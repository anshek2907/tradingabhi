
from signal_list import update_signal_list

# Mock some telegram lines
telegram_lines = ["15:30 EURUSD CALL"]

print("Testing signal loading...")
# Trigger update
updated_list = update_signal_list(telegram_lines)

print(f"Total signals in list: {len(updated_list)}")
for s in updated_list:
    time_str = s['time'].strftime('%H:%M')
    is_gen = "(GENERATED)" in str(s) # Not quite right since it's a dict, but let's check raw_line if it was stored
    # Actually raw_line isn't in the state dict, but I can check the time
    print(f"Signal: {time_str} {s['direction']}")

# Check if any generated signals are present (comparing against what I saw in generated_signals.json)
# I saw 17:05 CALL, 19:40 CALL, etc.
gen_found = any(s['time'].strftime('%H:%M') == "17:05" for s in updated_list)
print(f"Generated signal (17:05) found: {gen_found}")
