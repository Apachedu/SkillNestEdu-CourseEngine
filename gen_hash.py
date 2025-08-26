import hashlib

for pwd in ["Arushi!13","Skillnest2025!"]:
    h = hashlib.sha256(("skillnest-salt-v1"+pwd).encode()).hexdigest()
    print(pwd, "→", h)
