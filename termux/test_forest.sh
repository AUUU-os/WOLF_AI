#!/bin/bash
# WOLF_AI Forest Test - S24 Ultra Termux
# Run this to verify your installation works

echo "
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║   🐺 WOLF_AI Forest Test - S24 Ultra Edition                     ║
║   ════════════════════════════════════════════                    ║
║                                                                   ║
║   Testing your installation...                                    ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PASSED=0
FAILED=0

# Test function
test_check() {
    local name="$1"
    local cmd="$2"

    echo -n "Testing: $name... "
    if eval "$cmd" > /dev/null 2>&1; then
        echo -e "${GREEN}PASSED${NC}"
        ((PASSED++))
    else
        echo -e "${RED}FAILED${NC}"
        ((FAILED++))
    fi
}

# Core tests
echo ""
echo "=== Core System ==="
test_check "Python 3" "python3 --version"
test_check "pip" "pip --version"
test_check "git" "git --version"

# WOLF_AI tests
echo ""
echo "=== WOLF_AI Installation ==="
test_check "WOLF_AI directory" "[ -d ~/WOLF_AI ]"
test_check "config.py" "[ -f ~/WOLF_AI/config.py ]"
test_check "core/alpha.py" "[ -f ~/WOLF_AI/core/alpha.py ]"
test_check "core/wolf.py" "[ -f ~/WOLF_AI/core/wolf.py ]"
test_check "core/pack.py" "[ -f ~/WOLF_AI/core/pack.py ]"

# Modules tests
echo ""
echo "=== Modules ==="
test_check "modules/hunt.py" "[ -f ~/WOLF_AI/modules/hunt.py ]"
test_check "modules/track.py" "[ -f ~/WOLF_AI/modules/track.py ]"
test_check "modules/howl.py" "[ -f ~/WOLF_AI/modules/howl.py ]"
test_check "modules/sandbox.py" "[ -f ~/WOLF_AI/modules/sandbox.py ]"

# Termux tests
echo ""
echo "=== Termux Bridge ==="
test_check "termux_bridge.py" "[ -f ~/WOLF_AI/termux/termux_bridge.py ]"
test_check "termux-api (notifications)" "command -v termux-notification"
test_check "termux-api (toast)" "command -v termux-toast"

# Voice tests
echo ""
echo "=== Voice Control ==="
test_check "voice_control.py" "[ -f ~/WOLF_AI/voice/voice_control.py ]"
test_check "termux-tts-speak" "command -v termux-tts-speak"
test_check "termux-speech-to-text" "command -v termux-speech-to-text"

# Python imports test
echo ""
echo "=== Python Imports ==="

cd ~/WOLF_AI

test_check "Import config" "python3 -c 'from config import WOLF_ROOT; print(WOLF_ROOT)'"
test_check "Import core.wolf" "python3 -c 'from core.wolf import Wolf, Alpha'"
test_check "Import core.pack" "python3 -c 'from core.pack import Pack, get_pack'"
test_check "Import FastAPI" "python3 -c 'from fastapi import FastAPI'"
test_check "Import httpx" "python3 -c 'import httpx'"

# Optional tests
echo ""
echo "=== Optional Components ==="
test_check "Anthropic SDK" "python3 -c 'import anthropic' 2>/dev/null" || echo -e "${YELLOW}  (Install: pip install anthropic)${NC}"
test_check "OpenAI SDK" "python3 -c 'import openai' 2>/dev/null" || echo -e "${YELLOW}  (Install: pip install openai)${NC}"

# Summary
echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""
echo -e "Results: ${GREEN}$PASSED passed${NC}, ${RED}$FAILED failed${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed! Your forest is ready.${NC}"
    echo ""
    echo "Quick start commands:"
    echo "  wolf-server    - Start API server"
    echo "  wolf-status    - Check pack status"
    echo "  wolf-voice     - Start voice control"
    echo ""
    echo "AUUUUUUUUUUUUUUUUUU! 🐺"
else
    echo -e "${YELLOW}⚠ Some tests failed. Check the output above.${NC}"
    echo ""
    echo "Common fixes:"
    echo "  pip install -r requirements.txt"
    echo "  pkg install termux-api"
    echo ""
fi
