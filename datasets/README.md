# Datasets Overview

This directory contains datasets related to known exploited vulnerabilities, vulnerability intelligence, and exploit/tool coverage.

## Directory Structure

### `datasets/kev/`

Contains yearly datasets derived from the Known Exploited Vulnerabilities (KEV) catalog.

Files:
- `kev_2020.json`
- `kev_2021.json`
- `kev_2022.json`
- `kev_2023.json`
- `kev_2024.json`

Each file includes the CVE identifiers of vulnerabilities that were actively exploited in the corresponding year.

---

### `datasets/metasploit-nuclei/`

Contains yearly datasets mapping vulnerabilities to publicly available offensive security tools and detection templates.

Files:
- `metasploit-nuclei_2020.json`
- `metasploit-nuclei_2021.json`
- `metasploit-nuclei_2022.json`
- `metasploit-nuclei_2023.json`
- `metasploit-nuclei_2024.json`

Each file includes the CVE identifiers of vulnerabilities associated with Metasploit modules and Nuclei templates.

---

### `datasets/nvd/`

Contains the main yearly vulnerability datasets extracted from the NVD (National Vulnerability Database).

Files:
- `nvd_2020.json`
- `nvd_2021.json`
- `nvd_2022.json`
- `nvd_2023.json`
- `nvd_2024.json`

Each dataset contains vulnerability records and related metadata, which may include:
- CVE identifiers
- CVSS metrics
- CWE classifications
- Affected vendors/products
- Publication dates
- Additional enrichment data