from pathlib import Path
from pydantic import BaseModel
import json

class Vault(BaseModel):
    path: Path

def get_vaults_from_file(file_path: Path) -> list[Vault]:
    if not file_path.exists():
        file_path.write_text("[]")
        return []

    vaults = json.loads(file_path.read_text())

    return [
        Vault.model_validate(v)
        for v in vaults
    ]

def add_vault_to_file(file_path: Path, vault: Vault) -> None:
    vaults = get_vaults_from_file(file_path)
    vaults.append(vault)

    vaults = [
        v.model_dump_json()
        for v in vaults
    ]

    file_path.write_text(json.dumps(vaults, indent=2))

def search_vault_for_md_files(vault_path: Path) -> list[Path]:
    return list(vault_path.rglob("*.md"))