#!/bin/bash
# WOLF_AI Pack Skill for OpenClaw
# Usage: wolf <command> [args]

WOLF_API_URL="${WOLF_API_URL:-http://localhost:8000}"
WOLF_API_KEY="${WOLF_API_KEY:-}"

# Helper function for API calls
wolf_api() {
    local method=$1
    local endpoint=$2
    local data=$3

    if [ -n "$data" ]; then
        curl -s -X "$method" \
            -H "Content-Type: application/json" \
            -H "X-API-Key: $WOLF_API_KEY" \
            -d "$data" \
            "${WOLF_API_URL}${endpoint}"
    else
        curl -s -X "$method" \
            -H "X-API-Key: $WOLF_API_KEY" \
            "${WOLF_API_URL}${endpoint}"
    fi
}

# Parse command
command=$1
shift

case "$command" in
    status)
        echo "🐺 Getting pack status..."
        wolf_api GET "/api/status" | jq -r '
            "Pack Status: \(.pack.pack_status // "unknown")\n" +
            "Wolves:\n" +
            ((.pack.wolves // {}) | to_entries | map("  - \(.key): \(.value.status // "unknown")") | join("\n"))
        ' 2>/dev/null || wolf_api GET "/api/status"
        ;;

    awaken)
        echo "🐺 Awakening the pack..."
        wolf_api POST "/api/awaken" | jq -r '.message // .status // .' 2>/dev/null || wolf_api POST "/api/awaken"
        echo "AUUUUUUUUUUUUUUUUUU!"
        ;;

    howl)
        message="$1"
        frequency="medium"

        # Parse --frequency flag
        while [[ $# -gt 0 ]]; do
            case $1 in
                --frequency|-f)
                    frequency="$2"
                    shift 2
                    ;;
                *)
                    if [ -z "$message" ]; then
                        message="$1"
                    fi
                    shift
                    ;;
            esac
        done

        if [ -z "$message" ]; then
            echo "Usage: wolf howl \"message\" [--frequency low|medium|high|AUUUU]"
            exit 1
        fi

        echo "🐺 Sending howl..."
        wolf_api POST "/api/howl" "{\"message\": \"$message\", \"frequency\": \"$frequency\"}"
        echo ""
        ;;

    hunt)
        target="$1"
        assigned_to="hunter"

        # Parse --assign flag
        while [[ $# -gt 0 ]]; do
            case $1 in
                --assign|-a)
                    assigned_to="$2"
                    shift 2
                    ;;
                *)
                    if [ -z "$target" ]; then
                        target="$1"
                    fi
                    shift
                    ;;
            esac
        done

        if [ -z "$target" ]; then
            echo "Usage: wolf hunt \"target\" [--assign wolf_name]"
            exit 1
        fi

        echo "🐺 Starting hunt: $target"
        wolf_api POST "/api/hunt" "{\"target\": \"$target\", \"assigned_to\": \"$assigned_to\"}"
        echo ""
        ;;

    wilk)
        question="$1"
        mode="chat"

        # Parse --mode flag
        while [[ $# -gt 0 ]]; do
            case $1 in
                --mode|-m)
                    mode="$2"
                    shift 2
                    ;;
                *)
                    if [ -z "$question" ]; then
                        question="$1"
                    fi
                    shift
                    ;;
            esac
        done

        if [ -z "$question" ]; then
            echo "Usage: wolf wilk \"question\" [--mode chat|hacker|hustler|bro|guardian]"
            exit 1
        fi

        echo "🐺 Asking WILK ($mode mode)..."
        response=$(wolf_api POST "/api/wilk" "{\"message\": \"$question\", \"mode\": \"$mode\"}")
        echo "$response" | jq -r '.response // .' 2>/dev/null || echo "$response"
        ;;

    howls)
        limit="${1:-10}"
        echo "🐺 Recent howls:"
        wolf_api GET "/api/howls?limit=$limit" | jq -r '
            .howls[]? | "[\(.timestamp[11:16] // "?")] \(.from // "?"): \(.howl // "?")"
        ' 2>/dev/null || wolf_api GET "/api/howls?limit=$limit"
        ;;

    sync)
        echo "🐺 Syncing with GitHub..."
        wolf_api POST "/api/sync" | jq -r '.output // .status // .' 2>/dev/null || wolf_api POST "/api/sync"
        ;;

    *)
        echo "🐺 WOLF_AI Pack Control"
        echo ""
        echo "Commands:"
        echo "  wolf status              - Get pack status"
        echo "  wolf awaken              - Awaken the pack"
        echo "  wolf howl \"msg\" [-f freq] - Send howl (low/medium/high/AUUUU)"
        echo "  wolf hunt \"target\" [-a wolf] - Start a hunt"
        echo "  wolf wilk \"question\" [-m mode] - Ask WILK AI"
        echo "  wolf howls [limit]       - Get recent howls"
        echo "  wolf sync                - Sync with GitHub"
        echo ""
        echo "AUUUUUUUUUUUUUUUUUU!"
        ;;
esac
