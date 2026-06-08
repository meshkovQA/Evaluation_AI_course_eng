"""
Interactive CLI chat for testing multi-turn conversations
with the ability to save and evaluate conversations (DeepEval / Opik)
"""

from agent_connector import AgentConnector
from conversation_storage import ConversationStorage

# Import from evaluators module
from evaluators import DEFAULT_CHATBOT_ROLE
from evaluators.deepeval import run_evaluation as run_deepeval_evaluation
from evaluators.opik import run_evaluation as run_opik_evaluation


# ============= CONFIGURATION =============

# Agent settings
AGENT_CONFIG = {
    "endpoint_url": "http://5.11.83.110:8005/ask",
    "api_key": "80456142-5441-4469-b97f-1d72b7802a93",
    "user_id": "AleksM"
}

# ============= EVALUATION MODEL SETTINGS =============
# Model name (uses standard OPENAI_API_KEY env var)
MODEL_NAME = "gpt-4o-mini"

# Threshold for metrics
EVALUATION_THRESHOLD = 0.5

# Default framework: "deepeval" or "opik"
DEFAULT_FRAMEWORK = "deepeval"

# Project name for Opik
OPIK_PROJECT_NAME = "multiturn-evaluation"


def print_help():
    """Prints command reference"""
    print("\n" + "=" * 60)
    print("📋 COMMANDS:")
    print("=" * 60)
    print("  /new         - save current conversation and start a new one")
    print("  /eval        - evaluate conversations (current framework)")
    print("  /eval deepeval - evaluate with DeepEval")
    print("  /eval opik   - evaluate with Opik")
    print("  /framework   - show/change evaluation framework")
    print("  /list        - show list of saved conversations")
    print("  /clear       - delete all saved conversations")
    print("  /delete <id> - delete conversation by ID")
    print("  /url <url>   - add URL for next request")
    print("  /urls        - show current URLs")
    print("  /clearurls   - clear URLs")
    print("  /help        - show this help")
    print("  /exit        - exit the program")
    print("=" * 60)


def print_banner():
    """Prints banner on startup"""
    print("\n" + "=" * 60)
    print("🤖 MULTITURN CHAT EVALUATOR")
    print("   Interactive chat with multi-turn conversation evaluation")
    print("   Supports: DeepEval & Opik")
    print("=" * 60)
    print("\nType /help for a list of commands")
    print("-" * 60)


def run_eval_with_framework(conversations, framework: str, threshold: float):
    """Starts evaluation with the selected framework"""

    print(f"   Model: {MODEL_NAME} (OpenAI)")

    if framework == "deepeval":
        run_deepeval_evaluation(
            conversations=conversations,
            model=MODEL_NAME,
            threshold=threshold,
            chatbot_role=DEFAULT_CHATBOT_ROLE
        )
    elif framework == "opik":
        run_opik_evaluation(
            conversations=conversations,
            model=MODEL_NAME,
            verbose=True,
            project_name=OPIK_PROJECT_NAME,
        )
    else:
        print(f"❌ Unknown framework: {framework}")


def main():
    """Main function — interactive CLI chat"""

    print_banner()

    # Initialize components
    agent = AgentConnector(**AGENT_CONFIG)
    storage = ConversationStorage()

    # State
    current_conversation = []
    current_urls = []
    current_framework = DEFAULT_FRAMEWORK

    print(f"\n🔗 Agent: {AGENT_CONFIG['endpoint_url']}")
    print(f"📁 Saved conversations: {storage.count()}")
    print(f"🆔 Session: {agent.session_id}")
    print(f"📊 Evaluation framework: {current_framework}")

    while True:
        try:
            # User input
            user_input = input("\n\n👤 You: ").strip()

            if not user_input:
                continue

            # ============= COMMANDS =============

            if user_input == "/help":
                print_help()
                continue

            elif user_input == "/new":
                # Save current conversation and start a new one
                if current_conversation:
                    conv_id = storage.save_conversation(current_conversation)
                    print(f"\n✅ Conversation saved: {conv_id}")
                    print(f"   Total conversations: {storage.count()}")
                else:
                    print("\n⚠️  Current conversation is empty")

                current_conversation = []
                current_urls = []
                agent.new_session()
                print(f"🔄 New conversation started (session: {agent.session_id})")
                continue

            elif user_input.startswith("/eval"):
                # Determine framework
                parts = user_input.split()
                if len(parts) > 1:
                    framework = parts[1].lower()
                else:
                    framework = current_framework

                # Save current if exists
                if current_conversation:
                    save = input(
                        "Save current conversation before evaluation? (y/n): ").strip().lower()
                    if save == 'y':
                        conv_id = storage.save_conversation(
                            current_conversation)
                        print(f"✅ Conversation saved: {conv_id}")

                conversations = storage.load_all()
                if not conversations:
                    print("\n⚠️  No saved conversations to evaluate")
                    continue

                print(
                    f"\n📊 Evaluating {len(conversations)} conversations ({framework})...")

                run_eval_with_framework(
                    conversations=conversations,
                    framework=framework,
                    threshold=EVALUATION_THRESHOLD
                )
                continue

            elif user_input.startswith("/framework"):
                parts = user_input.split()
                if len(parts) > 1:
                    new_framework = parts[1].lower()
                    if new_framework in ["deepeval", "opik"]:
                        current_framework = new_framework
                        print(f"✅ Framework changed to: {current_framework}")
                    else:
                        print(f"❌ Unknown framework: {new_framework}")
                        print("   Available: deepeval, opik")
                else:
                    print(f"\n📊 Current framework: {current_framework}")
                    print("   To change: /framework deepeval or /framework opik")
                continue

            elif user_input == "/list":
                storage.list_conversations()
                continue

            elif user_input == "/clear":
                confirm = input(
                    "Delete ALL saved conversations? (yes/no): ").strip().lower()
                if confirm == 'yes':
                    storage.clear()
                    print("✅ All conversations deleted")
                else:
                    print("❌ Cancelled")
                continue

            elif user_input.startswith("/delete "):
                conv_id = user_input[8:].strip()
                if storage.delete(conv_id):
                    print(f"✅ Conversation {conv_id} deleted")
                else:
                    print(f"❌ Conversation {conv_id} not found")
                continue

            elif user_input.startswith("/url "):
                url = user_input[5:].strip()
                if url:
                    current_urls.append(url)
                    print(f"✅ URL added: {url}")
                    print(
                        f"   Current URLs ({len(current_urls)}): {current_urls}")
                continue

            elif user_input == "/urls":
                if current_urls:
                    print(f"\n🔗 Current URLs ({len(current_urls)}):")
                    for i, url in enumerate(current_urls, 1):
                        print(f"   {i}. {url}")
                else:
                    print("\n⚠️  No URLs set")
                continue

            elif user_input == "/clearurls":
                current_urls = []
                print("✅ URLs cleared")
                continue

            elif user_input == "/exit":
                # Offer to save current conversation
                if current_conversation:
                    save = input(
                        "Save current conversation before exiting? (y/n): ").strip().lower()
                    if save == 'y':
                        conv_id = storage.save_conversation(
                            current_conversation)
                        print(f"✅ Conversation saved: {conv_id}")

                print("\n👋 Goodbye!")
                break

            elif user_input.startswith("/"):
                print(f"❌ Unknown command: {user_input}")
                print("   Type /help for a list of commands")
                continue

            # ============= REGULAR MESSAGE =============

            # Show loading indicator
            print("\n⏳ Sending request...", end="", flush=True)

            # Send query to agent
            urls_to_send = current_urls if current_urls else None
            response = agent.query(user_input, urls=urls_to_send)

            # Clear URLs after use
            if current_urls:
                current_urls = []

            # Check for error
            if response.get('error'):
                print(f"\r❌ Error: {response['error']}")
                continue

            # Extract data
            answer = response.get('output', '')
            tools = response.get('tools_used', [])

            # Print answer
            print(f"\r🤖 Assistant: {answer}")

            if tools:
                tools_str = ', '.join(tools)
                print(f"\n   [🔧 Tools: {tools_str}]")

            # Add to current conversation
            current_conversation.append({
                "role": "user",
                "content": user_input
            })
            current_conversation.append({
                "role": "assistant",
                "content": answer,
                "tools_called": tools
            })

            # Show status
            turns_count = len(current_conversation) // 2
            print(
                f"\n   [💬 Turn {turns_count} | Unsaved messages: {len(current_conversation)}]")

        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user (Ctrl+C)")
            if current_conversation:
                save = input(
                    "Save current conversation? (y/n): ").strip().lower()
                if save == 'y':
                    conv_id = storage.save_conversation(current_conversation)
                    print(f"✅ Conversation saved: {conv_id}")
            print("👋 Goodbye!")
            break

        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
