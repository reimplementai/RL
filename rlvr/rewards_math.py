# reward functions for rlvr

import re, ast, operator
import math
import generate  # only used for colored printing in test

# Allowed: digits, + - * = and spaces
_ALLOWED_CHARS = set("0123456789+-*= ")
_NUM = set("0123456789")
# Very small, safe evaluator for +, -, * with integers only
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul}
_UN  = {ast.UAdd: lambda x: x, ast.USub: operator.neg}

def _safe_eval(expr: str) -> int:
    node = ast.parse(expr, mode="eval")
    def _eval(n):
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
            return _BIN[type(n.op)](_eval(n.left), _eval(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _UN:
            return _UN[type(n.op)](_eval(n.operand))
        if isinstance(n, ast.Constant) and isinstance(n.value, int):
            return n.value
        if isinstance(n, ast.Num):  # py<3.8 compatibility
            return int(n.n)
        raise ValueError("disallowed syntax")
    return _eval(node)

# Regex for an arithmetic expression like: [-]?d+ ([+*-] [-]?d+)*
_EXPR = r"\s*[+-]?\d+(?:\s*[+*-]\s*[+-]?\d+)*\s*"
_EQN_RE = re.compile(rf"^{_EXPR}={_EXPR}$")
_EXPR_RE = re.compile(rf"^{_EXPR}$")

low = 0.1
mid = low * 2.5
mid2 = 1.0
high = mid2 * 2.0 # 2.0
high2 = high * 1.25 # 2.5
highest = high2 * 1.4  # 3.5

# a weight between 0 (off scale) and 1 (very close)
def squash_log(delta):
    if delta < 0.1:
        return 1.15
    if delta < 0.5:
        return 1.03
    if delta < 1.0:
        return 1.02
    return 1 / (math.sqrt(delta) + 0.01)

def squash_lin(delta):
    if delta < 0.1:
        return 1.0
    return 1 / (math.sqrt(delta) + 0.01)

def reward_fn(s: str) -> float:
    s = s.strip()
    if not s:
        return 0.01

    # 1) Character check
    allowed_frac = sum(ch in _ALLOWED_CHARS for ch in s) / len(s)
    if allowed_frac < 1.0:
        # Contains illegal characters → low reward scaled by how “clean” it is
        return round(0.2 * (allowed_frac ** 2), 4)

    # 2) Equation case: A = B
    if _EQN_RE.match(s):
        left, right = map(str.strip, s.split("=", 1))
        try:
            lv = _safe_eval(left)
            rv = _safe_eval(right)
            #print(f"lv = {lv}, rv= {rv}")
        except Exception:
            return 0.39  # allowed chars but bad syntax/structure
        
        if lv == rv:
            return highest

        lv = lv + 1e-6
        rv = rv + 1e-6
        lin_delta = abs(lv-rv)
        log_delta = math.log(abs(lv)) - math.log(abs(rv))
        w_lin = squash_lin(lin_delta)
        w_log = squash_log(log_delta)
        w = (w_lin + w_log) / 2.0

        if lv < 10:
            r = mid + w_lin * (high - mid)
        else:  # prefer two digit outputs
            r = mid + w * (high - mid)
            if rv > 10:
                r = mid2 + w * (high2 - mid2)

        #print(f"{s}: \t lv={lv:.3f}, rv={rv:.3f}, linD={lin_delta:.3f}, logD={log_delta:.3f}, w_lin={w_lin:.3f}, w_log={w_log:.3f}, w={w:.3f}, r={r:.3f}")
        return r

    # 3) Expression-only case
    if _EXPR_RE.match(s):
        try:
            v = _safe_eval(s)
            scale = math.log(abs(v))
            if scale < 2:
                return 0.87
            if scale < 3:
                return 0.78
            if scale < 4:
                return 0.69
            return 0.61
        except Exception:
            return 0.57

    # 4) Allowed chars but malformed structure
    return 0.51

def reward_fn_hard(s: str) -> float:
    s = s.strip()
    if not s:
        return 0.01

    # 1) Character check
    allowed_frac = sum(ch in _ALLOWED_CHARS for ch in s) / len(s)
    if allowed_frac < 1.0:
        # Contains illegal characters → low reward scaled by how “clean” it is
        return round(0.2 * (allowed_frac ** 2), 4)

    # 2) Equation case: A = B
    if _EQN_RE.match(s):
        left, right = map(str.strip, s.split("=", 1))
        try:
            lv = _safe_eval(left)
            rv = _safe_eval(right)
            #print(f"lv = {lv}, rv= {rv}")
        except Exception:
            return 0.39  # allowed chars but bad syntax/structure
        
        if lv == rv:
            return highest

        lv = lv + 1e-6
        rv = rv + 1e-6
        lin_delta = abs(lv-rv)
        log_delta = math.log(abs(lv)) - math.log(abs(rv))
        w_lin = squash_lin(lin_delta)
        w_log = squash_log(log_delta)
        w = (100*w_lin + w_log) / 101.0

        r = mid + w * (high2 - mid)

        #print(f"{s}: \t lv={lv:.3f}, rv={rv:.3f}, linD={lin_delta:.3f}, logD={log_delta:.3f}, w_lin={w_lin:.3f}, w_log={w_log:.3f}, w={w:.3f}, r={r:.3f}")
        return r

    # 3) Expression-only case
    if _EXPR_RE.match(s):
        try:
            v = _safe_eval(s)
            scale = math.log(abs(v))
            if scale < 2:
                return 0.87
            if scale < 3:
                return 0.78
            if scale < 4:
                return 0.69
            return 0.61
        except Exception:
            return 0.57

    # 4) Allowed chars but malformed structure
    return 0.51

def reward_fn_hardest(s: str) -> float:
    s = s.strip()
    if not s:
        return 0.01

    # 1) Character check
    allowed_frac = sum(ch in _ALLOWED_CHARS for ch in s) / len(s)
    if allowed_frac < 1.0:
        # Contains illegal characters → low reward scaled by how “clean” it is
        return round(0.2 * (allowed_frac ** 2), 4)

    # 2) Equation case: A = B
    if _EQN_RE.match(s):
        left, right = map(str.strip, s.split("=", 1))
        try:
            lv = _safe_eval(left)
            rv = _safe_eval(right)
            #print(f"lv = {lv}, rv= {rv}")
        except Exception:
            return 0.39  # allowed chars but bad syntax/structure
        
        if lv == rv:
            return highest

        lv = lv + 1e-6
        rv = rv + 1e-6
        lin_delta = abs(lv-rv)
        log_delta = math.log(abs(lv)) - math.log(abs(rv))
        w_lin = squash_lin(lin_delta)
        w_log = squash_log(log_delta)
        w = (100*w_lin + w_log) / 101.0

        r = mid + w * (high - mid)

        #print(f"{s}: \t lv={lv:.3f}, rv={rv:.3f}, linD={lin_delta:.3f}, logD={log_delta:.3f}, w_lin={w_lin:.3f}, w_log={w_log:.3f}, w={w:.3f}, r={r:.3f}")
        return r

    # 3) Expression-only case
    if _EXPR_RE.match(s):
        try:
            v = _safe_eval(s)
            scale = math.log(abs(v))
            if scale < 2:
                return 0.87
            if scale < 3:
                return 0.78
            if scale < 4:
                return 0.69
            return 0.61
        except Exception:
            return 0.57

    # 4) Allowed chars but malformed structure
    return 0.51

def reward_fn_per_token(ids_ptext_outchars_plen):
    ids, ptext, outchars, plen = ids_ptext_outchars_plen
    outlen = len(outchars)
    reward_vector = [0.0] * outlen
    genlen = outlen - plen
    if genlen <= 0:
        return reward_vector

    left, right = map(str.strip, ptext.split("=", 1))
    try:
        lv = _safe_eval(left)
    except Exception:
        return reward_vector

    anschars = str(int(lv)) + "e" # EOS
    for k in range(genlen):
        i = plen+k
        if k < len(anschars):
            if outchars[i] == anschars[k]:
                reward_vector[i] += highest
            #elif outchars[i] in _NUM:
            #    if anschars[k] != ".":
            #        reward_vector[i] += 0.2
            #        ldigit = int(anschars[k])
            #        rdigit = int(outchars[i])
            #        diff = abs(ldigit - rdigit) + 0.1 # from 1 to 9
            #        reward_vector[i] += 1.0 / (diff*diff)
        
    #print(f"{ids_ptext_outchars_plen} -> {anschars} -> {reward_vector}")
    return reward_vector, anschars  # same size as input


prompts_gt_100 = []
prompts_lt_100 = []

answers = {}
for i in range(100):
    for j in range(100):
        ans = i+j
        a = str(ans)
        if ans >= 100:
            prompts_gt_100.append(f"{i}+{j}=")
        else:
            prompts_lt_100.append(f"{i}+{j}=")

#prompts = prompts_gt_100 + prompts_lt_100
prompts = prompts_lt_100
# split randomly into train and test:

import random
# set a deterministic seed for reproducibility
random.seed(42)
train_prompts = random.sample(prompts, int(len(prompts) * 0.8))
test_prompts = [p for p in prompts if p not in train_prompts]

if __name__ == "__main__":
    expressions = [
        "1+1=2",
        "12+9=21",
        "12+9=2 1",
        "1+2=2a",
        "3*2=6",
        "3*2=6a",
        "3*2=66",
        "6*9= 264 39 199- ",
        "0+5=48",
        "1+2",
        "11*990=898",
        "3+17=",
        "1+7=6268158",
        "2+14=5",
        "1+2=5",
        "72=7",
        "5*9=9",
        "1+5=5",
        "8*17=9",
        "13*15=1",
        "7+1=1",
        "16-18=1",
        "3*15=9",
        "3*15=10",
        "14-11=10",
        "16*7=50",
        "10+18=17",
        "13-11=1",
        "12+9=20",
        "1+5=4",
        "5*9=a",
        "5*9=",
        "5*9=1",
        "5*9=9",
        "5*9=10",
        "5*9=11",
        "5*9=100",
        "5*9=42",
        "5*9=43",
        "5*9=44",
        "5*9=46",
        "5*9=9*5",
        "5*9=45",
        "7+8=9",
        "7+8=29"
    ,]
    expressions = [
        "66+13=19",
        "66+13=119",

    ]

    arr = []
    for s in expressions:
        r = reward_fn_hard(s)
        arr.append((r, s))
    
    arr.sort()
    prev_r = 0.0
    for r, s in arr:
        print("  |  ")
        print(f"{r-prev_r:.2f}")
        print("  |  ")
        prev_r = r
        c = "green" if r > 1.0 else "red"
        r2 = reward_fn(s)
        generate.printcolor(f"{s} -> {r:.4f} | prev = {r2:.4f}", c)
    
    print(answers)
    print(f"low={low}, mid={mid}, mid2={mid2}, high={high}, high2={high2}, highest={highest}")

