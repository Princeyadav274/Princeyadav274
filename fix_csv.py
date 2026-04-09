import os

def fix_csv():
    file_path = 'data/bank.csv'
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.strip().split('\n')
    fixed_lines = []
    
    for line in lines:
        line = line.strip()
        if len(line) > 1 and line.startswith('"') and line.endswith('"'):
            # Strip outer quotes
            inner = line[1:-1]
            # Replace escaped "" with "
            inner = inner.replace('""', '"')
            fixed_lines.append(inner)
        else:
            fixed_lines.append(line)
            
    with open('data/bank_fixed.csv', 'w', encoding='utf-8') as f:
        f.write('\n'.join(fixed_lines) + '\n')
        
if __name__ == '__main__':
    fix_csv()
    print("Fixed CSV")
