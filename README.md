# Voting Panel Agent — Multi-Agent Spam Classifier

An ensemble multi-agent spam classification CLI system built in Python with **LangChain** and **LangGraph**, powered locally by **Ollama (`llama3.2`)**.

---

## 🏛️ Architecture Overview

The agent deploys three jurors with distinct personalities and instructions who independently evaluate incoming messages in parallel. A non-LLM aggregation node counts votes, determining whether a decisive majority exists or whether an impartial judge must break a tie.

```
                  ┌──────────────────────┐
                  │        START         │
                  └──────────┬───────────┘
                             │
            ┌────────────────┼────────────────┐
       FAN-OUT (Parallel Juror Evaluation)
            │                │                │
            ▼                ▼                ▼
     ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
     │   Strict    │  │  Friendly   │  │  Evidence   │
     │  Security   │  │  Reviewer   │  │   Analyst   │
     └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
            │                │                │
            └────────────────┼────────────────┘
                             │
                      FAN-IN (Tally)
                             │
                             ▼
                    ┌─────────────────┐
                    │      Tally      │ (NO LLM, ignores UNSURE)
                    └────────┬────────┘
                             │
                   Is there a majority?
                   (>= 2 equal votes)
                             │
               ┌─────────────┴─────────────┐
         Yes (verdict)               No (tie / split)
               │                           │
               ▼                           ▼
        ┌─────────────┐             ┌─────────────┐
        │     END     │             │    Judge    │ (LLM Tie-Breaker)
        └─────────────┘             └──────┬──────┘
                                           │
                                           ▼
                                    ┌─────────────┐
                                    │     END     │
                                    └─────────────┘
```

---

## 👥 Juror Personas & Prompts

| Juror | Role & Instruction | Input Data | Rule / Heuristic |
| :--- | :--- | :--- | :--- |
| **Strict Security Guard** (`strict`) | Distrusts everything. Votes SPAM if pushed to click, pay, hurry, or reply with personal details. | Original Message | Security-first skepticism |
| **Friendly Reviewer** (`friendly`) | Trusting person who gives benefit of doubt to normal chats, reminders, receipts. Obvious scams are still SPAM. | Original Message | Common sense generosity |
| **Data Analyst** (`evidence`) | Strict rules-based classifier. Reads counted heuristic signals and composite score. | Message + `spam_signals` | `score >= 4` → `SPAM`<br>`score <= 1` → `NOT_SPAM`<br>`score 2 or 3` → `UNSURE` |
| **Tie-Breaker Judge** (`judge`) | Only invoked when panel has no majority (split or unsure). Reviews the message and each juror's reason to make the final call. | Message + Juror Reasons | Binary decision (`SPAM` or `NOT_SPAM`) |

---

## 📁 Repository Structure

```
voting agent/
├── agent.py          # StateGraph definition, juror nodes, routing, and CLI loop
├── tools.py          # spam_signals(), parse_vote(), parse_reason(), and tally()
├── prompts.py        # System prompts for all 4 LLM roles
├── main.py           # CLI application entrypoint
├── requirements.txt  # Dependencies (langchain, langchain-ollama, langgraph)
├── test_suite.py     # 11 unit & end-to-end graph tests
├── prompt.md         # Reference prompt and specification
└── README.md         # Documentation and setup guide
```

---

## 🚀 Setup & Installation

### 1. Requirements
Ensure Python 3.10+ is installed:
```bash
pip install -r requirements.txt
```

### 2. Ollama Setup (Local LLM)
Make sure [Ollama](https://ollama.ai) is running with `llama3.2`:
```bash
ollama run llama3.2
```

*(Note: If Ollama is offline or unreachable, the agent includes an automatic persona-simulation fallback so you can test the graph logic end-to-end without disruption.)*

---

## 💻 Usage

### Option 1: Web Interface (Streamlit)
To launch the interactive browser UI:
```bash
streamlit run app.py
```
This opens `http://localhost:8501` in your browser with interactive juror cards, real-time warning signs breakdown, and preset test cases.

### Option 2: Terminal CLI
Run the command-line interface:
```bash
python main.py
```
Or directly:
```bash
python agent.py
```

### Example 1: Obvious Scam (Majority Vote)
```text
Message to check: URGENT! You won a FREE $1000 Walmart Gift Card! Click immediately: http://scam.me/win

Analyzing message with Voting Panel...

----------------------------------------
JUROR VOTES:
  Strict Security Guard : SPAM - Suspicious link, payment or urgency detected
  Friendly Reviewer     : SPAM - Obvious prize or money scam
  Data Analyst Evidence : SPAM - Calculated score is 10

----------------------------------------
TALLY: SPAM: 3

----------------------------------------
VERDICT: 🚫 SPAM (decided by majority vote)
================================================================
```

### Example 2: Normal Message (Majority Vote)
```text
Message to check: Hey Sarah, hope you have a nice day! Are we still meeting for lunch tomorrow at 1?

Analyzing message with Voting Panel...

----------------------------------------
JUROR VOTES:
  Strict Security Guard : NOT_SPAM - No obvious pressure or links detected
  Friendly Reviewer     : NOT_SPAM - Benefit of doubt given to reminder or chat
  Data Analyst Evidence : NOT_SPAM - Calculated score is 1

----------------------------------------
TALLY: NOT_SPAM: 3

----------------------------------------
VERDICT: ✅ NOT_SPAM (decided by majority vote)
================================================================
```

### Example 3: Borderline Split (Tie-Breaker Judge)
```text
Message to check: Urgent reminder: package delivery status at https://courier.com

Analyzing message with Voting Panel...

----------------------------------------
JUROR VOTES:
  Strict Security Guard : SPAM - Suspicious link, payment or urgency detected
  Friendly Reviewer     : NOT_SPAM - Benefit of doubt given to reminder or chat
  Data Analyst Evidence : UNSURE - Calculated score is 3

----------------------------------------
TALLY: SPAM: 1, NOT_SPAM: 1

----------------------------------------
VERDICT: 🚫 SPAM (decided by judge (tie-break))
================================================================
```

---

## 🧪 Running Tests

Execute the automated test suite covering signal extraction, vote parsing, tallying logic, and end-to-end graph execution:

```bash
python test_suite.py
```
