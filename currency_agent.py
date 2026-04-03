
import json
import os
import requests
from typing import TypedDict, Annotated

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "paste-key-here")

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

FREE_API_BASE = "https://open.er-api.com/v6/latest"


@tool
def get_exchange_rate(base_currency: str, target_currency: str) -> str:
    """
    Fetch the live exchange rate from base_currency to target_currency.

    Args:
        base_currency:   ISO 4217 code, e.g. 'USD', 'EUR', 'GBP'
        target_currency: ISO 4217 code, e.g. 'JPY', 'INR', 'CAD'

    Returns:
        JSON string with the rate and update timestamp, or an error.
    """
    base   = base_currency.upper().strip()
    target = target_currency.upper().strip()

    try:
        r    = requests.get(f"{FREE_API_BASE}/{base}", timeout=10)
        data = r.json()

        if data.get("result") != "success":
            return json.dumps({"error": f"Bad base currency '{base}': {data.get('error-type', 'unknown')}"})

        rates = data.get("rates", {})
        if target not in rates:
            return json.dumps({"error": f"Currency '{target}' not found.",
                               "hint": "Call list_supported_currencies to see all codes."})

        return json.dumps({
            "base_currency":   base,
            "target_currency": target,
            "rate":            rates[target],
            "last_updated":    data.get("time_last_update_utc", "N/A"),
        })

    except requests.RequestException as e:
        return json.dumps({"error": f"Network error: {e}"})


@tool
def convert_currency(amount: float, base_currency: str, target_currency: str) -> str:
    """
    Convert a specific amount from base_currency to target_currency.

    Args:
        amount:          Numeric amount to convert, e.g. 100.0
        base_currency:   ISO 4217 source currency code, e.g. 'USD'
        target_currency: ISO 4217 target currency code, e.g. 'EUR'

    Returns:
        JSON string with the converted amount and rate used, or an error.
    """
    base   = base_currency.upper().strip()
    target = target_currency.upper().strip()

    try:
        r    = requests.get(f"{FREE_API_BASE}/{base}", timeout=10)
        data = r.json()

        if data.get("result") != "success":
            return json.dumps({"error": f"Bad base currency '{base}': {data.get('error-type', 'unknown')}"})

        rates = data.get("rates", {})
        if target not in rates:
            return json.dumps({"error": f"Currency '{target}' not found."})

        rate      = rates[target]
        converted = round(amount * rate, 4)

        return json.dumps({
            "original_amount":  amount,
            "base_currency":    base,
            "target_currency":  target,
            "exchange_rate":    rate,
            "converted_amount": converted,
            "last_updated":     data.get("time_last_update_utc", "N/A"),
        })

    except requests.RequestException as e:
        return json.dumps({"error": f"Network error: {e}"})


@tool
def list_supported_currencies() -> str:
    """
    Return all currency codes supported by the exchange rate API.

    Returns:
        JSON string with total count and sorted list of ISO 4217 codes.
    """
    try:
        r    = requests.get(f"{FREE_API_BASE}/USD", timeout=10)
        data = r.json()

        if data.get("result") != "success":
            return json.dumps({"error": "Could not fetch currency list."})

        currencies = sorted(data.get("rates", {}).keys())
        return json.dumps({"total": len(currencies), "currencies": currencies})

    except requests.RequestException as e:
        return json.dumps({"error": f"Network error: {e}"})


TOOLS    = [get_exchange_rate, convert_currency, list_supported_currencies]
TOOL_MAP = {t.name: t for t in TOOLS}

llm = ChatGroq(
    model="llama-3.3-70b-versatile",  # free, fast, excellent at tool use
    temperature=0,
    api_key=GROQ_API_KEY,
).bind_tools(TOOLS)

SYSTEM_PROMPT = """You are an expert currency conversion assistant with access to live exchange rates.

Your tools:
- get_exchange_rate        → look up the rate between two currencies
- convert_currency         → convert an amount from one currency to another
- list_supported_currencies → show all available currency codes

Rules:
- Always use ISO 4217 codes (USD, EUR, JPY, GBP, INR, etc.)
- Never guess or recall exchange rates from memory — always call a tool
- Show converted amounts rounded to 2 decimal places
- Mention when the rate was last updated so users know the data is fresh
- If a currency is ambiguous, ask the user to clarify
"""

def agent_node(state: AgentState) -> AgentState:
    """Invoke Claude with the full conversation so far."""
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


def tool_node(state: AgentState) -> AgentState:
    """Execute every tool call the model requested."""
    last    = state["messages"][-1]
    results = []

    for tc in last.tool_calls:
        name   = tc["name"]
        args   = tc["args"]
        tc_id  = tc["id"]
        output = TOOL_MAP[name].invoke(args) if name in TOOL_MAP else \
                 json.dumps({"error": f"Unknown tool: {name}"})
        results.append(ToolMessage(content=output, tool_call_id=tc_id))

    return {"messages": results}

def should_use_tools(state: AgentState) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("agent", agent_node)
    g.add_node("tools", tool_node)
    g.set_entry_point("agent")
    g.add_conditional_edges("agent", should_use_tools, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile()


app = build_graph()

def run_query(query: str) -> str:
    """Single-turn query — returns the agent's final answer as a string."""
    result = app.invoke({"messages": [HumanMessage(content=query)]})
    return result["messages"][-1].content


def chat():
    """Interactive multi-turn CLI chat."""
    print("\n" + "=" * 55)
    print("  💱  Currency Agent  •  Claude + LangGraph")
    print("  Live rates: open.er-api.com  •  type 'exit' to quit")
    print("=" * 55 + "\n")

    history: list[BaseMessage] = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!"); break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "q"}:
            print("Goodbye!"); break

        history.append(HumanMessage(content=user_input))
        result  = app.invoke({"messages": history})
        history = result["messages"]
        print(f"\nAgent: {history[-1].content}\n")

def run_demo():
    queries = [
        "What is the current exchange rate from USD to EUR?",
        "Convert 250 British Pounds to Japanese Yen",
        "How much is 1000 Indian Rupees in Canadian Dollars?",
        "What currencies do you support?",
    ]
    print("\n" + "=" * 55)
    print("  💱  Currency Agent — Demo")
    print("=" * 55)
    for q in queries:
        print(f"\n❓ {q}")
        print(f"💬 {run_query(q)}")
        print("-" * 55)

if __name__ == "__main__":
    import sys
    run_demo() if "--demo" in sys.argv else chat()