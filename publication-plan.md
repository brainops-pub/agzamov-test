# Agzamov Test — Publication & Launch Plan

**BrainOps Limited · February 2026**

---

## Goal

Make the Agzamov Test a recognized standard benchmark for AI memory infrastructure evaluation.

---

## Steps

### Step 1: arXiv Preprint
- Reformat paper to LaTeX (NeurIPS or ICML template)
- Get endorsement for cs.AI or cs.LG category
- Submit → 1-3 day moderation → live with DOI
- Priority date established

### Step 2: First Results
- Run MVP (Phase A): 500 chess games, Claude with/without BrainOps memory
- Calculate Δₐ, τ, p-value
- Insert results into paper before arXiv submission
- Paper with data >> paper without data

### Step 3: GitHub Repo
- Open source test harness (Apache 2.0 or MIT)
- README links to arXiv paper
- Reproducible: clone, configure API keys, run
- Publish simultaneously with arXiv

### Step 4: Leaderboard
- HuggingFace Spaces (free) or agzamovtest.com
- Interactive table: model × memory × game format → Δₐ, τ
- Anyone can submit results
- Launch 1 week after arXiv

### Step 5: Outreach
- Twitter/X post with key visualization (win rate curve + τ)
- Reddit: r/MachineLearning, r/LocalLLaMA
- Direct outreach to memory infrastructure developers:
  - sgx-labs/statelessagent
  - eidetic-works/nucleus-mcp
  - zircote/subcog
  - mehdig-dev/shabka
  - 0xK3vin/MegaMemory
  - gnufoo/MeCP
- Contact Simon Willison (blog covers LLM tooling)
- HackerNews submission

### Step 6: Conference (optional, 6-12 months)
- NeurIPS, ICML, or AAAI — workshop paper or poster
- Peer review adds academic weight
- Not required for industry adoption

### Step 7: Trademark
- Register "Agzamov Test" via IPONZ (New Zealand)
- Optional: USPTO (United States)
- Cost: ~$300-500 NZD (NZ), ~$250-350 USD (US)
- Do when traction established, not urgent

---

## Timeline

| Step | Target | Duration | Dependency |
|------|--------|----------|------------|
| MVP code | Week 1-3 | 2-3 weeks | Developer assigned |
| First results | Week 4 | 1 week | API costs ~$50-100 |
| LaTeX paper | Week 4-5 | 3-5 days | Results ready |
| arXiv submission | Week 5 | 1 day | Endorsement secured |
| GitHub repo | Week 5 | Same day as arXiv | — |
| Leaderboard | Week 6 | 1 week | — |
| Outreach | Week 5-6 | Ongoing | Paper live |
| Conference | Month 6-12 | Optional | — |
| Trademark | When ready | 2-3 months processing | — |

---

## Key Blocker

**arXiv endorsement.** First-time submitters need endorsement from someone who has published in cs.AI or cs.LG. 

Options:
1. Contact NZ-based ML researchers (University of Otago, University of Canterbury, University of Auckland)
2. Contact authors of related projects on GitHub
3. Use alternative platforms first (OpenReview, Semantic Scholar) while seeking endorsement
4. Post on r/MachineLearning asking for endorsement (common practice)

---

## API Grants & Research Credits

### Strategy

The Agzamov Test is inherently valuable to model providers — if their model scores well, it's free marketing. Pitch: "We're building an open benchmark measuring how well your model uses external tools. Results published open-source with your model prominently featured."

### Anthropic

**AI for Science Program**
- Up to $20,000 API credits for 6 months
- Applications reviewed first Monday of each month
- Focus: scientific research — our benchmark qualifies as model evaluation research
- Form: https://www.anthropic.com/ai-for-science-program-rules

**External Researcher Access Program**
- Free API credits for AI safety/alignment research
- Model evaluation = core safety topic
- Form: https://support.claude.com/en/articles/9125743

**Anthropic Fellows Program**
- Next cohorts: May and July 2026
- 4 months, $2,100/week stipend + ~$10,000/month compute
- Includes "adversarial robustness" as research area
- Full-time commitment required

### OpenAI

**Researcher Access Program**
- Up to $1,000 API credits, 12 months validity
- Quarterly review: March, June, September, December
- Next review: March 2026 — submit NOW
- Form: https://openai.smapply.org/prog/openai_researcher_access_program/

### Google

**Google for Startups Gemini Kit**
- Free Gemini API credits + tools for startups
- BrainOps Limited qualifies as startup
- No formal application — get API key directly

**Google Academic Research Awards**
- Up to $100K USD — requires university PI (not directly applicable)
- Potential path: partner with NZ university researcher

### Application Plan

1. **March 2026:** Submit to OpenAI Researcher Access + Anthropic AI for Science simultaneously
2. **March 2026:** Get Google Gemini API key via Startups Kit
3. **If approved:** Full 3×3 matrix at zero cost
4. **Estimated total credits if all approve:** ~$21,000+ in API access

---

## Budget Estimate

| Item | Cost |
|------|------|
| API calls for MVP (500 chess games) | ~$50-100 USD |
| API calls for full matrix (9 combinations × 2 games) | ~$500-1000 USD |
| Domain agzamovtest.com | ~$15 USD/year |
| IPONZ trademark | ~$300-500 NZD |
| arXiv | Free |
| HuggingFace Spaces | Free |
| GitHub | Free |

Total to first publication: **under $200 USD**
Total for full matrix + leaderboard: **under $1500 USD**

---

*BrainOps Limited · Agzamov Test*
