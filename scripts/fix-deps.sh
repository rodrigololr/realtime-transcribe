#!/usr/bin/env bash
# Captions — consertador automático de dependências do sistema.
#
# Detecta PPAs quebrados (404 / sem Release / arch mismatch) que impedem
# `apt update` de completar, desabilita-os renomeando para .disabled,
# e instala os pacotes necessários pro captions rodar.
#
# Uso:  ./scripts/fix-deps.sh [--dry-run]
#       ./scripts/fix-deps.sh --dry-run    # só mostra o que faria
#
# Não remove PPAs — só desabilita. Reversível (basta renomear de volta).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

REQUIRED_PACKAGES=(
    "pulseaudio-utils"     # provê `parec`
    "libpulse-dev"         # headers (necessário pra algumas builds)
    "python3-tk"           # Tkinter pro overlay GUI
)

log()  { printf '[fix-deps] %s\n' "$*"; }
warn() { printf '[fix-deps] AVISO: %s\n' "$*" >&2; }
err()  { printf '[fix-deps] ERRO: %s\n' "$*" >&2; }

require_root() {
    if [[ $EUID -ne 0 ]]; then
        err "precisa rodar como root. Use: sudo $0"
        exit 1
    fi
}

# ----------------------------------------------------------------------
# 1. Detecta PPAs quebrados rodando `apt-get update` e capturando warnings
# ----------------------------------------------------------------------
detect_broken_sources() {
    # Estratégia em 2 passos:
    #   1. Identifica QUAIS sources falharam (linhas de erro/warning do apt)
    #   2. Mapeia cada uma → arquivo .list correspondente

    local output
    output="$(apt-get update 2>&1)" || output="$(apt-get update 2>&1)" || true

    # Passo 1: extrai URLs apenas das linhas de erro.
    # Heurística: linha é erro se começa com "E:", "W:", "Erro:", "Err:",
    # contém "404", "does not have", "não tem", "não tem suporte",
    # ou "Release" + algo errado.
    local broken_urls=()
    while IFS= read -r url; do
        broken_urls+=("$url")
    done < <(printf '%s\n' "$output" \
        | grep -E '^(E|W|Erro|Err):| 404 |does not have a Release|não tem|não tem suporte' \
        | grep -oE 'https?://[^ ]+' \
        | sort -u || true)

    if [[ ${#broken_urls[@]} -eq 0 ]]; then
        return 0
    fi

    # Passo 2: pra cada URL quebrada, acha o .list
    local sources_dir="/etc/apt/sources.list.d"
    local matches=()
    local url
    for url in "${broken_urls[@]}"; do
        # Normaliza: tira scheme, colapsa /, tira /ubuntu trailing
        local norm
        norm="$(printf '%s' "$url" | sed -E 's|^https?://||; s|/+|/|g; s|/$||; s|/ubuntu$||')"
        for file in "$sources_dir"/*.list; do
            [[ -f "$file" ]] || continue
            if grep -qF "$norm" "$file" 2>/dev/null; then
                matches+=("$file")
            fi
        done
    done

    # Deduplica e imprime
    if [[ ${#matches[@]} -gt 0 ]]; then
        printf '%s\n' "${matches[@]}" | sort -u
    fi
}

# ----------------------------------------------------------------------
# 2. Desabilita (renomeia) os .list quebrados
# ----------------------------------------------------------------------
disable_sources() {
    local broken_files
    mapfile -t broken_files < <(detect_broken_sources)

    if [[ ${#broken_files[@]} -eq 0 ]]; then
        log "nenhum PPA quebrado encontrado"
        return 0
    fi

    log "PPAs quebrados detectados:"
    local f
    for f in "${broken_files[@]}"; do
        log "  - $(basename "$f")"
    done

    log ""
    if $DRY_RUN; then
        log "(dry-run) não vou modificar nada"
        log "Para aplicar de verdade, rode sem --dry-run"
        return 0
    fi

    log "desabilitando..."
    for f in "${broken_files[@]}"; do
        local target="${f}.disabled"
        if [[ -e "$target" ]]; then
            warn "$(basename "$f") já tem .disabled — pulando"
            continue
        fi
        if mv -- "$f" "$target"; then
            log "  $(basename "$f") → $(basename "$target")"
        else
            err "  falhou ao renomear $f"
            return 1
        fi
    done
    log ""
}

# ----------------------------------------------------------------------
# 3. apt update limpo + install dos pacotes necessários
# ----------------------------------------------------------------------
install_packages() {
    log "rodando apt update..."
    if ! apt-get update; then
        err "apt update falhou mesmo após desabilitar PPAs. Veja saída acima."
        return 1
    fi

    log ""
    log "instalando pacotes necessários: ${REQUIRED_PACKAGES[*]}"
    if $DRY_RUN; then
        log "(dry-run) não vou instalar nada"
        return 0
    fi

    if ! DEBIAN_FRONTEND=noninteractive apt-get install -y "${REQUIRED_PACKAGES[@]}"; then
        err "apt install falhou"
        return 1
    fi
    log ""
}

# ----------------------------------------------------------------------
# 4. Verifica que `parec` ficou disponível
# ----------------------------------------------------------------------
verify() {
    log "verificando..."
    local missing=()
    for pkg in "${REQUIRED_PACKAGES[@]}"; do
        if ! dpkg -s "$pkg" >/dev/null 2>&1; then
            missing+=("$pkg")
        fi
    done

    if [[ ${#missing[@]} -gt 0 ]]; then
        err "ainda faltam pacotes: ${missing[*]}"
        return 1
    fi

    if ! command -v parec >/dev/null 2>&1; then
        err "parec não está no PATH mesmo com pulseaudio-utils instalado"
        return 1
    fi

    log ""
    log "tudo OK! parec está em: $(command -v parec)"
    log "agora rode: ./run.sh"
}

main() {
    require_root

    if $DRY_RUN; then
        log "modo dry-run — nenhuma alteração será feita"
        log ""
    fi

    disable_sources
    install_packages
    verify
}

main "$@"
