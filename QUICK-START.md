# Quick Start: TL;DR for Impatient People

**TL;DR**: You have a cloud/DevOps background. Most of what you need is in Week 1. Skip heavy prep. Start Monday.

---

## Can You Do This?

Answer honestly (30 seconds):

1. Write a Python FastAPI endpoint? **YES** → Continue
2. Use Docker? **YES** → Continue  
3. Understand JSON/YAML? **YES** → Continue
4. Never touched LLMs before? **YES** → OK, read "Minimal Crash Course" below

---

## Minimal Crash Course (30 min)

**If you know 0 about LLMs, spend 30 minutes here:**

1. **Tokens** (5 min): Go to https://platform.openai.com/tokenizer. Paste text. See token count. Get the idea: "hello world" = ~2 tokens, longer text = more tokens = more cost.

2. **Embeddings** (10 min): Text → vector of 1536 numbers. Similar texts have similar vectors. That's it. Used for semantic search.

3. **RAG** (10 min): 
   - User asks: "What is X?"
   - System retrieves relevant docs
   - System asks LLM: "Answer this using these docs"
   - LLM answers
   - Done. That's RAG.

4. **Fine-tuning** (5 min): Adjust a pre-trained model for your specific task. Expensive ($$$). Only use if RAG doesn't work.

---

## What You Already Know (Don't Re-Learn)

- **Python APIs**: You can build this
- **Docker**: You know this
- **Deployment thinking**: You got it
- **Git/GitHub**: You know this
- **Linux/DevOps**: Your strength

You only need to learn: **LLM concepts** (tokens, embeddings, RAG, prompts).

---

## Prerequisites You Can Skip

- [ ] Deep ML/AI theory (you don't need it)
- [ ] Math (backprop, linear algebra—nope)
- [ ] CS fundamentals (you have them)
- [ ] LangChain deep-dive (you'll learn in Week 1)

---

## Prerequisites You MUST Have

- [x] Comfortable with Python APIs
- [x] Docker basics
- [x] Git basics
- [x] Can read code without panic
- [ ] **Understand: tokens, embeddings, RAG** (30 min crash course above)

---

## Two Paths

### Path A: Just Start (Recommended for You)

**Monday morning:**
1. Read Week 1 Day 1 learning section (30 min)
2. Start coding (RAG ingestion)
3. Learn gaps as you hit them

**Pro**: Faster, more motivating, learn-by-doing  
**Con**: May need to re-read things (acceptable)  
**Time**: Smooth into Week 1 immediately

### Path B: Prep First (If You're Cautious)

**This week:**
1. Minimal crash course above (30 min)
2. Read LEARNING-SPRINT.md sections 4–6 (6 hours)
3. Run code examples

**Then:** Start Week 1 with confidence  
**Time**: 2–3 days part-time

---

## What NOT to Do

❌ Don't read academic papers on transformers  
❌ Don't take a 4-week online course  
❌ Don't memorize tokenizer implementations  
❌ Don't worry about being "not ready"  

---

## Your Week 0 (Optional)

If you choose Path B, here's your schedule:

**Tuesday (2 hours)**
- Read: [OpenAI token guide](https://platform.openai.com/docs/guides/tokens) (20 min)
- Do: Count tokens on 10 texts (20 min)
- Watch: [RAG explainer (5 min YouTube)](https://www.youtube.com/results?search_query=rag+explained+5+minutes)

**Wednesday (2 hours)**
- Read: [Pinecone Vector Databases 101](https://www.pinecone.io/learn/vector-database/) (45 min)
- Do: Generate embeddings + compute similarity (75 min, see LEARNING-SPRINT.md Part 5)

**Thursday (2 hours)**
- Read: [LlamaIndex RAG concepts](https://docs.llamaindex.ai/en/stable/getting_started/concepts/) (30 min)
- Do: Sketch RAG pipeline (30 min)
- Rest: You're ready for Monday

**Friday–Sunday**: Rest, plan Week 1 setup

---

## Week 1 Expectation

**You'll build a working RAG system in 5 days** (15.5 hours of focused work).

Yes, you're learning LLM concepts on the fly. That's intentional.

Your strength (building, architecture, systems) will carry you. LLM-specific stuff you'll pick up fast.

---

## Red Flags (Stop & Prep If These Apply)

🚩 "I've never written Python code"  
→ Pause. Spend 1 week on Python basics first.

🚩 "I don't understand APIs"  
→ Pause. Build a simple FastAPI app first.

🚩 "I'm uncomfortable with command line"  
→ Pause. Get comfortable with bash/git first.

🚩 "I have zero LLM experience and want depth first"  
→ OK, do LEARNING-SPRINT.md section-by-section (24 hours). Then Week 1.

---

## Go/No-Go Checklist

Before Week 1, confirm:

- [ ] Can write a Python function
- [ ] Can use `pip` to install packages
- [ ] Know what an API is
- [ ] Comfortable with `git`
- [ ] Know roughly what embeddings are (from 30-min crash course)
- [ ] Know roughly what RAG is (from 30-min crash course)

**All YES?** → Start Week 1 Monday. 🚀

**1–2 NO?** → Spend 4 hours on those, then start.

**3+ NO?** → Do PREREQUISITES.md or LEARNING-SPRINT.md first.

---

## Week 1 Day 1 Plan

**Monday (3.5 hours)**

Morning (1.5 hours):
- [ ] Read: Tokens + context window (30 min)
- [ ] Read: RAG pipeline (30 min)
- [ ] Question: "Why RAG, not fine-tuning?" (write 3 sentences)

Afternoon (2 hours):
- [ ] Create Python project structure
- [ ] Write first module: `ingestion.py`
- [ ] Test it on sample document

**End of Day 1**: You have environment working, first code written, LLM concepts clear.

---

## Resource Links (Save These)

**Must-Have (Bookmark)**
- [OpenAI Tokenizer](https://platform.openai.com/tokenizer) — Interactive
- [OpenAI API Docs](https://platform.openai.com/docs) — Reference
- [LangChain Docs](https://python.langchain.com/docs) — Your framework

**Good-to-Have**
- [LlamaIndex Docs](https://docs.llamaindex.ai) — Alternative framework
- [Pinecone Vector DB 101](https://www.pinecone.io/learn/vector-database/) — Background reading

**Specific to Week 1–4**
- See each WEEK-X-CHECKLIST.md for links

---

## FAQ

**Q: Do I need to understand transformers?**  
A: No. Know that LLMs generate text token-by-token. That's enough.

**Q: Will I really ship a RAG system in Week 1?**  
A: Yes. Basic RAG (ingest → embed → retrieve → generate). Running on your laptop.

**Q: How much AI knowledge do I really need?**  
A: Enough to explain: "RAG retrieves docs, then asks LLM to answer using those docs." That's the core.

**Q: What if I get stuck?**  
A: Week 1 Day 1 learning section is designed for this. Re-read it. Try the code examples.

**Q: Can I do this part-time?**  
A: Yes. Week 1 = 15.5 hours. That's 3 hours/day for 5 days, or spread across 2 weeks at 1.5 hours/day.

---

## Decision Time

**Pick one:**

1. **Skip prep, start Week 1 Monday** (30 min crash course first)
   - Speed: Fast
   - Risk: May need to re-read things
   - Motivation: Immediate building

2. **Spend 2–3 days on prep, start Week 1 next week** (Week 0 schedule above)
   - Speed: Relaxed
   - Risk: Low
   - Motivation: Confident entry

3. **Deep dive prerequisites** (LEARNING-SPRINT.md, 24 hours)
   - Speed: Slowest
   - Risk: Lowest
   - Motivation: Mastery mindset

---

**My recommendation for your profile**: Pick #1. You're a builder. Build. Learn as you go.

---

## What's Next?

1. **Do the 30-min crash course** (above)
2. **Create repo structure** (see Week 1 Day 1)
3. **Monday 9am, start Week 1**
4. **Ship working RAG by Friday**

---

**You got this. Let's build.** 🚀

#LearnInPublic #BuildInPublic

---

**Last Updated**: 2025-12-28
