import time
from rich.console import Console
from rich.panel import Panel
from rich.theme import Theme

# Batman-inspired Dark Theme
batman_theme = Theme({
    "info": "dim white",
    "warning": "bold yellow",
    "danger": "bold red",
    "system": "bold bright_black on bright_white",
    "tutor": "bold yellow",
    "thought": "dim bright_black",
    "citation": "italic bright_black"
})

console = Console(theme=batman_theme)

def run_demo():
    console.clear()
    welcome_msg = """
[tutor]🦇 SOCRATIC TUTOR PROTOCOL INITIATED[/tutor]
[info]System:[/info] Adaptive RAG Pipeline Online
[info]Directive:[/info] To guide, not to give.
    """
    console.print(Panel(welcome_msg, border_style="yellow", title="[bold yellow]WAYNE ENTERPRISES[/bold yellow]"))

    console.print("[info]Connection established. Awaiting input... (Type 'quit' to exit)[/info]\n")
    
    # Mocking Student Input visually
    console.print("[bold white]Student:[/bold white] What exactly is chunking in Retrieval-Augmented Generation?")
    
    with console.status("[dim bright_black]Synthesizing hybrid vectors & formulating Socratic vector...[/dim bright_black]"):
        time.sleep(2)
    
    # Mocking Internal Thought Process
    thought_panel = Panel(
        "[thought]The student is asking about 'chunking'. I need to guide them to understand the core concept without revealing the exact definition. I will provide a comparative hint using real-world analogies like book indices or reading passages.[/thought]", 
        title="[bright_black]INTERNAL TELEMETRY (Thought Process)[/bright_black]", 
        border_style="bright_black",
        expand=False
    )
    console.print("\n", thought_panel)
    
    # Mocking Socratic Hint
    hint_panel = Panel(
        "That's a great question! Instead of giving you the exact definition, think about how an index works in a library or a glossary in a textbook. If you had to feed an entire 1,000-page book to a computer, it might be overwhelming. How do you think breaking it down relates to what you're asking about?", 
        title="[bold yellow]TUTOR[/bold yellow]", 
        border_style="yellow",
        expand=False
    )
    console.print(hint_panel)
    
    # Mocking Citations
    cit_text = "• [Course: Intro to AI | Ch: 4] Tags: [RAG, Vector DBs, Indexing]"
    cit_panel = Panel(
        f"[citation]{cit_text}[/citation]", 
        title="[bright_black]EXTRACTED DATA NODES[/bright_black]", 
        border_style="bright_black",
        expand=False
    )
    console.print(cit_panel)
    console.print("\n")

if __name__ == "__main__":
    run_demo()
