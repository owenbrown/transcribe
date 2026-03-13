"""Hardcoded reference data for development and testing.

Each record mirrors the schema of the address_references table:
vendor_name, address, city, postcode, country.
"""

REFERENCE_DATA: list[dict] = [
    # ── United States ──────────────────────────────────────────────
    {"vendor_name": "Apple Store", "address": "189 The Grove Dr", "city": "Los Angeles", "postcode": "90036", "country": "US"},
    {"vendor_name": "Starbucks", "address": "1912 Pike Pl", "city": "Seattle", "postcode": "98101", "country": "US"},
    {"vendor_name": "Walmart", "address": "5001 E Ray Rd", "city": "Phoenix", "postcode": "85044", "country": "US"},
    {"vendor_name": "Target", "address": "7100 Santa Monica Blvd", "city": "West Hollywood", "postcode": "90046", "country": "US"},
    {"vendor_name": "Whole Foods", "address": "4 Union Square S", "city": "New York", "postcode": "10003", "country": "US"},
    {"vendor_name": "Home Depot", "address": "3838 N Central Ave", "city": "Phoenix", "postcode": "85012", "country": "US"},
    {"vendor_name": "Best Buy", "address": "1015 N San Fernando Blvd", "city": "Burbank", "postcode": "91504", "country": "US"},
    # ── France ─────────────────────────────────────────────────────
    {"vendor_name": "Galeries Lafayette", "address": "40 Boulevard Haussmann", "city": "Paris", "postcode": "75009", "country": "FR"},
    {"vendor_name": "Carrefour", "address": "1 Rue Jean Mermoz", "city": "Paris", "postcode": "75008", "country": "FR"},
    {"vendor_name": "Fnac", "address": "136 Rue de Rennes", "city": "Paris", "postcode": "75006", "country": "FR"},
    {"vendor_name": "Boulangerie Paul", "address": "49 Rue de Rivoli", "city": "Paris", "postcode": "75001", "country": "FR"},
    {"vendor_name": "Monoprix", "address": "52 Avenue des Champs-Elysees", "city": "Paris", "postcode": "75008", "country": "FR"},
    {"vendor_name": "Decathlon", "address": "26 Avenue de la Grande Armee", "city": "Paris", "postcode": "75017", "country": "FR"},
    {"vendor_name": "Sephora", "address": "70 Avenue des Champs-Elysees", "city": "Paris", "postcode": "75008", "country": "FR"},
    # ── Germany ────────────────────────────────────────────────────
    {"vendor_name": "KaDeWe", "address": "Tauentzienstrasse 21-24", "city": "Berlin", "postcode": "10789", "country": "DE"},
    {"vendor_name": "Aldi", "address": "Brunnenstrasse 27", "city": "Berlin", "postcode": "10119", "country": "DE"},
    {"vendor_name": "Lidl", "address": "Skalitzer Strasse 80", "city": "Berlin", "postcode": "10997", "country": "DE"},
    {"vendor_name": "MediaMarkt", "address": "Alexanderplatz 1", "city": "Berlin", "postcode": "10178", "country": "DE"},
    {"vendor_name": "Rossmann", "address": "Friedrichstrasse 141", "city": "Berlin", "postcode": "10117", "country": "DE"},
    {"vendor_name": "dm", "address": "Kurfuerstendamm 227", "city": "Berlin", "postcode": "10719", "country": "DE"},
    {"vendor_name": "Edeka", "address": "Schoenhauser Allee 36", "city": "Berlin", "postcode": "10435", "country": "DE"},
    # ── Canada ─────────────────────────────────────────────────────
    {"vendor_name": "Tim Hortons", "address": "55 Bloor St W", "city": "Toronto", "postcode": "M4W 1A5", "country": "CA"},
    {"vendor_name": "Shoppers Drug Mart", "address": "700 Bay St", "city": "Toronto", "postcode": "M5G 1Z6", "country": "CA"},
    {"vendor_name": "Canadian Tire", "address": "839 Yonge St", "city": "Toronto", "postcode": "M4W 2H2", "country": "CA"},
    {"vendor_name": "Loblaws", "address": "60 Carlton St", "city": "Toronto", "postcode": "M5B 1J2", "country": "CA"},
    {"vendor_name": "Roots", "address": "100 Bloor St W", "city": "Toronto", "postcode": "M5S 1M5", "country": "CA"},
    {"vendor_name": "Hudson's Bay", "address": "176 Yonge St", "city": "Toronto", "postcode": "M5C 2L7", "country": "CA"},
    {"vendor_name": "MEC", "address": "300 Queen St W", "city": "Toronto", "postcode": "M5V 2A2", "country": "CA"},
]

# ── OCR test cases ────────────────────────────────────────────────
# Vendor name is always correct; address has realistic OCR corruption.
TEST_CASES: list[dict] = [
    {
        "vendor_name": "Apple Store",
        "ocr_address": "1B9 The Gr0ve Dr",
        "expected_address": "189 The Grove Dr",
        "expected_city": "Los Angeles",
        "description": "US: 8->B, o->0 OCR confusion",
    },
    {
        "vendor_name": "Galeries Lafayette",
        "ocr_address": "40 Bou1evard Haussrnann",
        "expected_address": "40 Boulevard Haussmann",
        "expected_city": "Paris",
        "description": "France: l->1, m->rn OCR confusion",
    },
    {
        "vendor_name": "KaDeWe",
        "ocr_address": "Tauentzienstra8e 2l-24",
        "expected_address": "Tauentzienstrasse 21-24",
        "expected_city": "Berlin",
        "description": "Germany: ss->8 (eszett-like), 1->l OCR confusion",
    },
    {
        "vendor_name": "Tim Hortons",
        "ocr_address": "55 B1oor St VV",
        "expected_address": "55 Bloor St W",
        "expected_city": "Toronto",
        "description": "Canada: l->1, W->VV OCR confusion",
    },
    {
        "vendor_name": "Walmart",
        "ocr_address": "5OO1 E Ray Rd",
        "expected_address": "5001 E Ray Rd",
        "expected_city": "Phoenix",
        "description": "US: 0->O OCR confusion",
    },
]
