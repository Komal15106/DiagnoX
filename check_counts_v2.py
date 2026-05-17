import re

def count_in_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        # Find the ALL_SYMPTOMS list
        match = re.search(r'ALL_SYMPTOMS\s*=\s*\[(.*?)\]', content, re.DOTALL)
        if match:
            items = re.findall(r'"([^"]+)"', match.group(1))
            return len(items)
    except Exception as e:
        return str(e)
    return 0

print(f"app.py: {count_in_file('app.py')}")
print(f"main.py: {count_in_file('main.py')}")
