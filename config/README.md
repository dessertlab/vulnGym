# Configuration Files
This directory contains configuration files used to model APT groups, targeted products, network exposure zones, and phishing susceptibility.
## Files Overview
### `apt_groups.json`
Contains information about Advanced Persistent Threat (APT) groups and their associated characteristics.
Each entry includes:
- `name` → Name of the APT group
- `mitre_url` → Reference to the MITRE ATT&CK group page
- `products` → List of targeted or commonly abused products/software
- `phishing` → Indicates whether the group commonly uses phishing techniques
- `goal` → Primary attack objective or operational goal

---

### `products.json`

Contains metadata associated with software products.
Each product entry includes:

* zones → Network exposure or deployment zones where the product is typically located
* phishing → Indicates whether the product is commonly associated with phishing-related attack vectors

Supported zones may include:

* external
* internal
* dmz
* database
---