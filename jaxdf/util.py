def update_dictionary(old, new_entries):
  for key, val in zip(new_entries.keys(), new_entries.values()):
    old[key] = val
  return old