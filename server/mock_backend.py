#
# Copyright (c) 2024–2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Mock backend data for the Flour & Frost cake-shop demo.

This is the file to edit when customizing the demo for your own hackathon
project: swap the catalog, add or remove "known customer" phone numbers, or
replace the dicts entirely with calls to a real backend (database, REST API,
etc.) from inside the tool functions in ``bot.py``.

Both lookups are case-insensitive on the key side in ``bot.py`` — cake names
are lowercased before lookup, and phone numbers should be stored in E.164
format (e.g. ``+14155551234``) to match Twilio's ``from_number``.

Each cake carries:
    price (USD), description, in_stock (bool), occasions (list of lowercase
    strings the LLM can filter on), on_special (bool — used by the "any
    deals?" path).
"""

CAKES = {
    "classic vanilla": {
        "price": 35.00,
        "description": "Vanilla sponge with vanilla buttercream",
        "in_stock": True,
        "occasions": ["birthday", "thank you", "just because", "celebration"],
        "on_special": False,
    },
    "double chocolate fudge": {
        "price": 42.00,
        "description": "Rich chocolate layers with fudge frosting",
        "in_stock": True,
        "occasions": ["birthday", "anniversary", "celebration", "indulgence"],
        "on_special": False,
    },
    "red velvet": {
        "price": 45.00,
        "description": "Red velvet layers with cream cheese frosting",
        "in_stock": True,
        "occasions": ["valentine's day", "anniversary", "romance", "birthday"],
        "on_special": True,
    },
    "lemon drizzle": {
        "price": 38.00,
        "description": "Zesty lemon cake with a citrus glaze",
        "in_stock": False,
        "occasions": ["spring", "afternoon tea", "thank you", "just because"],
        "on_special": False,
    },
    "carrot walnut": {
        "price": 40.00,
        "description": "Spiced carrot cake with cream cheese frosting",
        "in_stock": True,
        "occasions": ["birthday", "fall", "thanksgiving", "autumn"],
        "on_special": False,
    },
    "funfetti celebration": {
        "price": 44.00,
        "description": "Vanilla confetti cake with rainbow sprinkles",
        "in_stock": True,
        "occasions": ["birthday", "kids", "congratulations", "celebration"],
        "on_special": True,
    },
    "tiramisu torte": {
        "price": 52.00,
        "description": "Espresso-soaked layers with mascarpone cream",
        "in_stock": True,
        "occasions": ["anniversary", "dinner party", "indulgence", "date night"],
        "on_special": False,
    },
    "strawberry shortcake": {
        "price": 46.00,
        "description": "Vanilla sponge, fresh strawberries, and whipped cream",
        "in_stock": True,
        "occasions": ["spring", "summer", "birthday", "mother's day"],
        "on_special": False,
    },
    "wedding tier classic": {
        "price": 120.00,
        "description": "Three-tier vanilla and almond cake with white fondant",
        "in_stock": True,
        "occasions": ["wedding", "engagement", "anniversary"],
        "on_special": False,
    },
    "black forest": {
        "price": 50.00,
        "description": "Chocolate cake with cherries and whipped cream",
        "in_stock": False,
        "occasions": ["birthday", "christmas", "holiday", "celebration"],
        "on_special": False,
    },
    "salted caramel": {
        "price": 48.00,
        "description": "Caramel cake with salted caramel buttercream",
        "in_stock": True,
        "occasions": ["birthday", "congratulations", "indulgence", "thank you"],
        "on_special": True,
    },
    "matcha green tea": {
        "price": 47.00,
        "description": "Green tea sponge with white chocolate ganache",
        "in_stock": True,
        "occasions": ["afternoon tea", "spring", "just because"],
        "on_special": False,
    },
    "new baby celebration": {
        "price": 42.00,
        "description": "Soft vanilla cake with pastel buttercream",
        "in_stock": True,
        "occasions": ["new baby", "baby shower", "congratulations"],
        "on_special": False,
    },
    "graduation cap": {
        "price": 45.00,
        "description": "Chocolate cake decorated as a graduation cap",
        "in_stock": True,
        "occasions": ["graduation", "congratulations", "achievement"],
        "on_special": False,
    },
    "pumpkin spice": {
        "price": 40.00,
        "description": "Spiced pumpkin cake with maple frosting",
        "in_stock": True,
        "occasions": ["fall", "thanksgiving", "autumn", "halloween"],
        "on_special": True,
    },
}

# Add your own number here if you want to test the bot with a known customer
KNOWN_CUSTOMERS = {
    "+14155551234": {"name": "Alex", "last_order": "red velvet"},
    "+14155555678": {"name": "Jordan", "last_order": "classic vanilla"},
}
