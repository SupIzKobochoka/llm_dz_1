import ast
import json
import re
from collections import Counter


def schemas(system):
    decoder = json.JSONDecoder()
    for match in re.finditer(r'\[\s*\{', system):
        try:
            value, _ = decoder.raw_decode(system[match.start():])
        except ValueError:
            continue
        if isinstance(value, list) and value and isinstance(value[0], dict) and 'name' in value[0]:
            return [{'type': 'function', 'function': normalize(v)} for v in value]
    return []


def normalize(obj):
    obj = dict(obj)
    params = obj.get('parameters', {})

    def fix(v):
        if isinstance(v, list):
            return [fix(x) for x in v]
        if not isinstance(v, dict):
            return v
        out = {k: fix(x) for k, x in v.items()}
        if isinstance(out.get('type'), str):
            out['type'] = {
                'dict': 'object',
                'float': 'number',
                'tuple': 'array',
                'any': 'string',
            }.get(out['type'], out['type'])
        return out

    obj['parameters'] = fix(params)
    return obj


def parse_reference(s):
    s = s.strip()
    if not s.startswith('[') or not s.endswith(']'):
        return None

    body = s[1:-1].strip()
    if not body:
        return []

    calls = []
    parts, start, depth, quote, escape = [], 0, 0, None, False

    for i, c in enumerate(body):
        if quote:
            if escape:
                escape = False
            elif c == '\\':
                escape = True
            elif c == quote:
                quote = None
        elif c in ('"', "'"):
            quote = c
        elif c in '([{':
            depth += 1
        elif c in ')]}':
            depth -= 1
        elif c == ',' and depth == 0:
            parts.append(body[start:i].strip())
            start = i + 1

    parts.append(body[start:].strip())

    for part in parts:
        pos = part.find('(')
        if pos <= 0 or not part.endswith(')'):
            return None

        name, args = part[:pos].strip(), part[pos + 1:-1]
        if not name:
            return None

        try:
            node = ast.parse('f(' + args + ')', mode='eval').body
            if not isinstance(node, ast.Call) or node.args or any(k.arg is None for k in node.keywords):
                return None
            kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}
        except (ValueError, SyntaxError, TypeError, RecursionError):
            return None

        calls.append({'name': name, 'arguments': kwargs})

    return calls


def parse_prediction(text):
    text = text.strip()
    blocks = re.findall(r'<tool_call>(.*?)</tool_call>', text, re.S)

    if len(blocks) != text.count('<tool_call>') or len(blocks) != text.count('</tool_call>'):
        return None

    calls = []

    for block in blocks:
        functions = re.findall(r'<function=([^>]+)>(.*?)</function>', block, re.S)

        if len(functions) != 1:
            try:
                obj = json.loads(block.strip())
                name = obj.get('name')
                args = obj.get('arguments', {})

                if isinstance(args, str):
                    args = json.loads(args)

                if not isinstance(name, str) or not isinstance(args, dict):
                    return None

                calls.append({'name': name, 'arguments': args})
                continue
            except (ValueError, AttributeError, TypeError):
                return None

        name, inside = functions[0]
        pairs = re.findall(r'<parameter=([^>]+)>(.*?)</parameter>', inside, re.S)

        if len(pairs) != inside.count('<parameter=') or len(pairs) != inside.count('</parameter>'):
            return None

        args = {}

        for key, value in pairs:
            try:
                args[key] = json.loads(value.strip())
            except ValueError:
                args[key] = value.strip()

        calls.append({'name': name.strip(), 'arguments': args})

    if not calls and re.search(r'<(?:tool|function|parameter)', text):
        return None

    return calls


def _canonical(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
    )


def reward_one(prediction, target, mode='exact'):
    calls = parse_prediction(prediction)
    if calls is None:
        return 0.0

    expected = json.loads(target) if isinstance(target, str) else target

    if not expected:
        return 1.0 if not calls and prediction.strip() else 0.0

    if not calls:
        return 0.0

    gold = Counter(_canonical(c) for c in expected)
    actual = Counter(_canonical(c) for c in calls)

    if mode == 'exact':
        return float(gold == actual)

    names_gold = Counter(c['name'] for c in expected)
    names_pred = Counter(c['name'] for c in calls)
    name_hits = sum((names_gold & names_pred).values())
    exact_hits = sum((gold & actual).values())
    denom = max(len(expected), len(calls))

    return 0.2 + 0.3 * name_hits / denom + 0.5 * exact_hits / denom


def reward_fn(completions, target, mode='exact', **kwargs):
    return [
        reward_one(c if isinstance(c, str) else str(c), t, mode)
        for c, t in zip(completions, target)
    ]