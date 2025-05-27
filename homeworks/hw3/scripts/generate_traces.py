#!/usr/bin/env python3
"""Generate Recipe Bot traces for dietary adherence evaluation.

This script sends dietary preference queries to the Recipe Bot and collects
the responses to create a dataset for LLM-as-Judge evaluation.
"""

import sys
import os
import pandas as pd
import random
from pathlib import Path
from typing import List, Dict, Any
from rich.console import Console
from rich.progress import track
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from rich.text import Text
from rich.panel import Panel
from rich.markdown import Markdown
from rich.console import Group

# Add the backend to the path so we can import the Recipe Bot
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.utils import get_agent_response

MAX_WORKERS = 32

console = Console()

def load_dietary_queries(csv_path: str) -> List[Dict[str, Any]]:
    """Load dietary preference queries from CSV file."""
    df = pd.read_csv(csv_path)
    return df.to_dict('records')

def generate_trace(query: str, dietary_restriction: str) -> Dict[str, Any]:
    """Generate a single Recipe Bot trace for a dietary query."""
    try:
        # Create the conversation with just the user query
        messages = [{"role": "user", "content": query}]
        
        # Get the bot's response
        updated_messages = get_agent_response(messages)
        
        # Extract the assistant's response
        assistant_response = updated_messages[-1]["content"]
        
        return {
            "query": query,
            "dietary_restriction": dietary_restriction,
            "response": assistant_response,
            "success": True,
            "error": None
        }
    except Exception as e:
        console.print(f"[red]Error generating trace for query: {query}")
        console.print(f"[red]Error: {str(e)}")
        return {
            "query": query,
            "dietary_restriction": dietary_restriction,
            "response": None,
            "success": False,
            "error": str(e)
        }

def generate_trace_with_id(args: tuple) -> Dict[str, Any]:
    """Wrapper function for parallel processing."""
    query_data, trace_num = args
    query = query_data["query"]
    dietary_restriction = query_data["dietary_restriction"]
    
    trace = generate_trace(query, dietary_restriction)
    trace["trace_id"] = f"{query_data['id']}_{trace_num}"
    trace["query_id"] = query_data["id"]
    return trace

def generate_multiple_traces_per_query(queries: List[Dict[str, Any]], 
                                     traces_per_query: int = 40,
                                     max_workers: int = MAX_WORKERS) -> List[Dict[str, Any]]:
    """Generate multiple traces for each query using parallel processing."""
    
    # Create all the tasks
    tasks = []
    for query_data in queries:
        for i in range(traces_per_query):
            tasks.append((query_data, i + 1))
    
    all_traces = []
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {executor.submit(generate_trace_with_id, task): task for task in tasks}
        
        # Process completed tasks with progress tracking
        with console.status("[yellow]Generating traces in parallel...") as status:
            completed = 0
            total = len(tasks)
            
            for future in as_completed(future_to_task):
                trace = future.result()
                all_traces.append(trace)
                completed += 1
                
                if completed % 100 == 0:
                    # Display the trace just generated for verification
                    panel_content = Text()
                    panel_content.append(f"Trace ID: {trace['trace_id']}\n", style="bold magenta")
                    panel_content.append(f"Query ID: {trace['query_id']}\n", style="bold cyan")
                    panel_content.append(f"Dietary Restriction: {trace['dietary_restriction']}\n", style="bold yellow")
                    panel_content.append(f"Success: {trace['success']}\n", style="bold green" if trace['success'] else "bold red")
                    panel_content.append("Query:\n", style="bold blue")
                    panel_content.append(f"{trace['query']}\n\n")
                    
                    if trace['success'] and trace['response']:
                        response_markdown = Markdown(trace['response'])
                        panel_group = Group(
                            panel_content,
                            Markdown("--- Response ---"),
                            response_markdown
                        )
                    else:
                        error_text = Text(f"Error: {trace.get('error', 'Unknown error')}", style="bold red")
                        panel_group = Group(panel_content, error_text)
                    
                    console.print(Panel(
                        panel_group,
                        title="Sample Generated Trace",
                        border_style="cyan"
                    ))
                
                status.update(f"[yellow]Generated {completed}/{total} traces ({completed/total*100:.1f}%)")
    
    
    console.print(f"[green]Completed parallel generation of {len(all_traces)} traces")
    return all_traces

def save_traces(traces: List[Dict[str, Any]], output_path: str) -> None:
    """Save traces to CSV file."""
    df = pd.DataFrame(traces)
    df.to_csv(output_path, index=False)
    console.print(f"[green]Saved {len(traces)} traces to {output_path}")

def main():
    """Main function to generate Recipe Bot traces."""
    console.print("[bold blue]Recipe Bot Trace Generation")
    console.print("=" * 50)
    
    # Set up paths
    script_dir = Path(__file__).parent
    hw3_dir = script_dir.parent
    data_dir = hw3_dir / "data"
    
    # Load dietary queries
    queries_path = data_dir / "dietary_queries.csv"
    if not queries_path.exists():
        console.print(f"[red]Error: {queries_path} not found!")
        return
    
    queries = load_dietary_queries(str(queries_path))
    console.print(f"[green]Loaded {len(queries)} dietary queries")
    
    # Generate traces (40 traces per query)
    console.print("[yellow]Generating traces... This may take a while.")
    traces = generate_multiple_traces_per_query(queries, traces_per_query=40)
    
    # Filter successful traces
    successful_traces = [t for t in traces if t["success"]]
    failed_traces = [t for t in traces if not t["success"]]
    
    console.print(f"[green]Successfully generated {len(successful_traces)} traces")
    if failed_traces:
        console.print(f"[yellow]Failed to generate {len(failed_traces)} traces")
    
    # Save traces
    output_path = data_dir / "raw_traces.csv"
    save_traces(successful_traces, str(output_path))
    
    # Print summary statistics
    console.print("\n[bold]Summary Statistics:")
    console.print(f"Total traces generated: {len(successful_traces)}")
    
    # Count by dietary restriction
    restriction_counts = {}
    for trace in successful_traces:
        restriction = trace["dietary_restriction"]
        restriction_counts[restriction] = restriction_counts.get(restriction, 0) + 1
    
    console.print("\nTraces per dietary restriction:")
    for restriction, count in sorted(restriction_counts.items()):
        console.print(f"  {restriction}: {count}")

if __name__ == "__main__":
    main() 