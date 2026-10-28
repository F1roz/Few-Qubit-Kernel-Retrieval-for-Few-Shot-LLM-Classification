"""Phase 6: prompt, call the LLM, cache, parse. Owner: LLM role.
Providers: anthropic, openai (paid APIs), hf_local (free open model on the GPU), mock (free fake for tests)."""
import json, hashlib, time
from pathlib import Path


def build_prompt(query_text, example_texts, example_labels, sp, order="near_last", seed=0):
    """sp = datasets_registry.spec(name). Examples come ordered MOST similar first.
    order = near_last (default, used in every main run): print least similar first, so the most similar
            example sits right above the query. Same rule for every selector.
    order = near_first or shuffled: only for the ordering ablation."""
    pairs = list(zip(example_texts, example_labels))
    if order == "near_last":
        pairs = list(reversed(pairs))
    elif order == "shuffled":
        import random
        random.Random(seed).shuffle(pairs)
    parts = [sp["instruction"], ""]
    for t, l in pairs:
        parts += [f"{sp['item']}: {t}", f"Answer: {l}", ""]
    parts += [f"{sp['item']}: {query_text}", "Answer:"]
    return "\n".join(parts)


def parse_answer(raw, labels):
    """Exact label first, then the LONGEST label the reply starts with (so 'card_payment_fee_charged' is not read
    as a shorter label), then, for multi-word labels written with spaces or cut off by the token limit, the one
    label the reply is a clear prefix of. Anything else is invalid."""
    a = str(raw).strip().split("\n")[0].strip().lower().strip(".:;,!'\"*` ")
    forms = [a, a.replace(" ", "_").replace("-", "_")]
    for f in forms:
        if f in labels:
            return f
    for f in forms:
        hits = [l for l in labels if f.startswith(l) and (len(f) == len(l) or not f[len(l)].isalpha())]
        if hits:
            return max(hits, key=len)
    for f in forms:
        hits = [l for l in labels if len(f) >= 10 and l.startswith(f)]
        if len(hits) == 1:
            return hits[0]
    return "invalid"


class LLM:
    def __init__(self, provider, model, temperature=0, max_tokens=24, cache_file="results/llm_cache.jsonl",
                 load_in_4bit=False):
        self.provider, self.model, self.load_in_4bit = provider, model, load_in_4bit
        self.temperature, self.max_tokens = temperature, max_tokens
        self.cache_file = Path(cache_file); self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache = {}
        if self.cache_file.exists():
            for line in self.cache_file.read_text().splitlines():
                r = json.loads(line); self.cache[r["key"]] = r["answer"]

    def _key(self, prompt):
        return hashlib.sha256((self.provider + "|" + self.model + f"|{self.max_tokens}||" + prompt).encode()).hexdigest()

    def _load_hf(self):
        if not hasattr(self, "_tok"):
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self._torch = torch
            self._tok = AutoTokenizer.from_pretrained(self.model)
            if self.load_in_4bit:                        # 7B model on a 16 GB T4: 4-bit weights (bitsandbytes)
                from transformers import BitsAndBytesConfig
                q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
                self._lm = AutoModelForCausalLM.from_pretrained(self.model, quantization_config=q, device_map="auto").eval()
            else:
                self._lm = AutoModelForCausalLM.from_pretrained(
                    self.model, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
                self._lm.to("cuda" if torch.cuda.is_available() else "cpu").eval()

    def _call(self, prompt):
        if self.provider == "mock":                       # free fake: copies the most similar example's label
            answers = [l[8:] for l in prompt.splitlines() if l.startswith("Answer: ")]
            return answers[-1] if answers else "none"
        if self.provider == "anthropic":
            import anthropic
            r = anthropic.Anthropic().messages.create(model=self.model, max_tokens=self.max_tokens,
                                                      temperature=self.temperature,
                                                      messages=[{"role": "user", "content": prompt}])
            return r.content[0].text
        if self.provider == "openai":
            from openai import OpenAI
            r = OpenAI().chat.completions.create(model=self.model, temperature=self.temperature,
                                                 max_tokens=self.max_tokens,
                                                 messages=[{"role": "user", "content": prompt}])
            return r.choices[0].message.content
        if self.provider == "hf_local":                   # free open model, greedy decoding (same as temperature 0)
            self._load_hf()
            ids = self._tok.apply_chat_template([{"role": "user", "content": prompt}],
                                                add_generation_prompt=True, return_tensors="pt").to(self._lm.device)
            with self._torch.no_grad():
                out = self._lm.generate(ids, max_new_tokens=self.max_tokens, do_sample=False,
                                        pad_token_id=self._tok.eos_token_id)
            return self._tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        raise ValueError(self.provider)

    def label_probs(self, prompt, labels):
        """Probability of each label word as the answer (needed for AUROC and calibration).
        hf_local: sum of the token log probabilities of each label after the prompt, then a softmax over labels.
        mock: a fake that puts 0.7 on the label the mock would answer. API providers return None."""
        key = "probs|" + self._key(prompt)
        if key in self.cache:
            return self.cache[key]
        if self.provider == "mock":                       # fake: rank weighted vote of the example labels in the prompt
            answers = [l[8:].strip() for l in prompt.splitlines() if l.startswith("Answer: ")]
            w = [0.3] * len(labels)
            for pos, ans in enumerate(answers):
                if ans in labels:
                    w[labels.index(ans)] += pos + 1
            p = [x / sum(w) for x in w]
        elif self.provider == "hf_local":
            self._load_hf(); torch = self._torch
            ctx = self._tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
            ctx_ids = self._tok(ctx, add_special_tokens=False).input_ids; n = len(ctx_ids)
            variants = [(j, v) for j, l in enumerate(labels) for v in {l, l.capitalize()}]   # "human" and "Human"
            seqs = [ctx_ids + self._tok(v, add_special_tokens=False).input_ids for _, v in variants]
            L = max(map(len, seqs)); pad = self._tok.pad_token_id or self._tok.eos_token_id
            ids = torch.tensor([q + [pad] * (L - len(q)) for q in seqs], device=self._lm.device)
            att = torch.tensor([[1] * len(q) + [0] * (L - len(q)) for q in seqs], device=self._lm.device)
            per_label = [[] for _ in labels]
            for c0 in range(0, len(seqs), 4):            # small batches; log-softmax only over the label positions
                with torch.no_grad():
                    logits = self._lm(input_ids=ids[c0:c0 + 4], attention_mask=att[c0:c0 + 4]).logits[:, n - 1:L - 1]
                lp = torch.log_softmax(logits.float(), -1)
                for b in range(lp.shape[0]):
                    (j, _), q = variants[c0 + b], seqs[c0 + b]
                    tok = ids[c0 + b, n:len(q)]
                    per_label[j].append(float(lp[b, :len(q) - n].gather(1, tok[:, None]).sum()))
            logps = [float(torch.logsumexp(torch.tensor(v), 0)) for v in per_label]
            m = max(logps); e = [pow(2.718281828459045, x - m) for x in logps]; p = [x / sum(e) for x in e]
        else:
            return None
        self.cache[key] = p
        with self.cache_file.open("a") as f:
            f.write(json.dumps({"key": key, "answer": p, "provider": self.provider, "model": self.model,
                                "ts": time.strftime("%Y-%m-%d"), "kind": "probs"}) + "\n")
        return p

    def ask(self, prompt):
        """Returns (answer, was_cached, latency_ms). Retries 3 times on errors."""
        key = self._key(prompt)
        if key in self.cache:
            return self.cache[key], True, 0
        for attempt in range(3):
            try:
                t0 = time.time(); ans = self._call(prompt); ms = int((time.time() - t0) * 1000)
                self.cache[key] = ans
                with self.cache_file.open("a") as f:   # model and time are kept for the paper (model name, access dates)
                    f.write(json.dumps({"key": key, "answer": ans, "provider": self.provider, "model": self.model,
                                        "ts": time.strftime("%Y-%m-%d"), "ms": ms}) + "\n")
                return ans, False, ms
            except Exception as e:
                print("LLM error, retrying:", e); time.sleep(2 * (attempt + 1))
        return "ERROR", False, -1
