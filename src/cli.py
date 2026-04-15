import os
import sys
from rich.console import Console
from rich.panel import Panel
from rich.theme import Theme
from rich.prompt import Prompt
from rich.markdown import Markdown
from src.socratic_graph import SocraticTutor

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

def display_welcome():
    console.clear()
    welcome_msg = """
[tutor]🦇 SOCRATIC TUTOR PROTOCOL INITIATED[/tutor]
[info]System:[/info] Adaptive RAG Pipeline Online
[info]Directive:[/info] To guide, not to give.
    """
    console.print(Panel(welcome_msg, border_style="yellow", title="[bold yellow]WAYNE ENTERPRISES[/bold yellow]"))

def format_citations(citations):
    if not citations:
        return ""
    
    cit_list = []
    for c in citations:
        course = c.get('course_name', 'Unknown Course')
        ch = c.get('chapter_number', 'N/A')
        tags = c.get('concept_tags', '')
        cit_list.append(f"• [Course: {course} | Ch: {ch}] Tags: [{tags}]")
        
    return "\n".join(cit_list)

def start_interactive_session():
    display_welcome()
    
    demo_mode = False
    if "COHERE_API_KEY" not in os.environ:
        console.print("[danger]WARNING: COHERE_API_KEY environment variable is not set.[/danger]")
        console.print("[warning]Initialization bypassing Cohere Cloud... Entering DEMO SIMULATION MODE.[/warning]")
        demo_mode = True
    else:    
        try:
            with console.status("[warning]Initializing Socratic State Graph and connecting to ChromaDB...[/warning]"):
                tutor = SocraticTutor()
        except Exception as e:
             console.print(f"[danger]Initialization Failed:[/danger] {e}")
             sys.exit(1)
         
    console.print("[info]Connection established. Awaiting input... (Type 'quit' to exit)[/info]\n")
    
    while True:
        try:
            student_query = Prompt.ask("[bold white]Student[/bold white]")
            
            if student_query.lower() in ['quit', 'exit', 'q']:
                console.print("\n[tutor]🦇 Terminating session. Good luck.[/tutor]")
                break
                
            if not student_query.strip():
                continue
                
            with console.status("[dim bright_black]Synthesizing hybrid vectors & formulating Socratic vector...[/dim bright_black]"):
                import time
                time.sleep(1.5) # simulate latency
                if demo_mode:
                    response = {
                        "internal_thought_process": f"The student is asking '{student_query}'. I need to guide them to understand the core concept without revealing the exact definition. I will provide a comparative hint.",
                        "socratic_hint": "That's a great question! Instead of giving you the exact definition, think about how an index works in a library or a glossary in a textbook. How do you think that concept relates to what you're asking about?",
                        "citations": [
                            {"course_name": "CS 101 - Intro to AI", "chapter_number": 4, "concept_tags": "Retrieval, Indexing"}
                        ]
                    }
                else:
                    response = tutor.run(student_query)
                
            # Render Internal Thought Process (Hidden technically, but displayed here for developer visibility)
            thought_panel = Panel(
                f"[thought]{response.get('internal_thought_process')}[/thought]", 
                title="[bright_black]INTERNAL TELEMETRY (Thought Process)[/bright_black]", 
                border_style="bright_black",
                expand=False
            )
            console.print("\n", thought_panel)
            
            # Render Socratic Hint
            hint_panel = Panel(
                response.get('socratic_hint'), 
                title="[bold yellow]TUTOR[/bold yellow]", 
                border_style="yellow",
                expand=False
            )
            console.print(hint_panel)
            
            # Render Citations if relevant
            citations = response.get('citations', [])
            if citations:
                cit_text = format_citations(citations)
                cit_panel = Panel(
                    f"[citation]{cit_text}[/citation]", 
                    title="[bright_black]EXTRACTED DATA NODES[/bright_black]", 
                    border_style="bright_black",
                    expand=False
                )
                console.print(cit_panel)
                
            console.print("\n")
            
        except KeyboardInterrupt:
            console.print("\n[warning]Operation aborted by user.[/warning]")
            break
        except Exception as e:
            console.print(f"\n[danger]Graph Execution Error:[/danger] {e}")

if __name__ == "__main__":
    start_interactive_session()
