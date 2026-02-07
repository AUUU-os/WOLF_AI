#!/data/data/com.termux/files/usr/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# WOLF_AI Termux Setup - Samsung S24 Ultra Edition
# ═══════════════════════════════════════════════════════════════════════════
#
# This script sets up WOLF_AI on Termux (Android)
# Your phone becomes the FOREST - the pack's command center
#
# Requirements:
# - Termux (F-Droid version recommended)
# - Termux:API (for device integration)
# - ~500MB storage
# - Internet connection
#
# Run: curl -sL https://raw.githubusercontent.com/AUUU-os/WOLF_AI/main/termux/setup.sh | bash
# Or:  bash setup.sh
#
# ═══════════════════════════════════════════════════════════════════════════

set -e

echo "
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║   🐺 WOLF_AI - Termux Setup                                               ║
║   ═══════════════════════════════════════                                 ║
║                                                                           ║
║   Samsung S24 Ultra = THE FOREST                                          ║
║   The pack's mobile command center                                        ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() { echo -e "${GREEN}[✓]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
error() { echo -e "${RED}[✗]${NC} $1"; exit 1; }
info() { echo -e "${BLUE}[i]${NC} $1"; }

# ═══════════════════════════════════════════════════════════════════════════
# Step 1: Update Termux packages
# ═══════════════════════════════════════════════════════════════════════════
info "Step 1/6: Updating Termux packages..."

pkg update -y && pkg upgrade -y
log "Packages updated"

# ═══════════════════════════════════════════════════════════════════════════
# Step 2: Install required packages
# ═══════════════════════════════════════════════════════════════════════════
info "Step 2/6: Installing required packages..."

pkg install -y \
    python \
    python-pip \
    git \
    nodejs \
    openssh \
    termux-api \
    jq \
    curl \
    wget \
    vim \
    htop

log "Packages installed"

# ═══════════════════════════════════════════════════════════════════════════
# Step 3: Setup storage access
# ═══════════════════════════════════════════════════════════════════════════
info "Step 3/6: Setting up storage access..."

if [ ! -d ~/storage ]; then
    termux-setup-storage
    sleep 2
fi
log "Storage access configured"

# ═══════════════════════════════════════════════════════════════════════════
# Step 4: Clone WOLF_AI
# ═══════════════════════════════════════════════════════════════════════════
info "Step 4/6: Cloning WOLF_AI repository..."

WOLF_DIR="$HOME/WOLF_AI"

if [ -d "$WOLF_DIR" ]; then
    warn "WOLF_AI already exists, pulling latest..."
    cd "$WOLF_DIR"
    git pull origin main || git pull origin claude/coding-task-7mwOW
else
    git clone https://github.com/AUUU-os/WOLF_AI.git "$WOLF_DIR"
    cd "$WOLF_DIR"
fi

log "WOLF_AI cloned to $WOLF_DIR"

# ═══════════════════════════════════════════════════════════════════════════
# Step 5: Install Python dependencies
# ═══════════════════════════════════════════════════════════════════════════
info "Step 5/6: Installing Python dependencies..."

cd "$WOLF_DIR"

# Upgrade pip
pip install --upgrade pip

# Install requirements (skip heavy optional deps for mobile)
pip install \
    fastapi \
    uvicorn \
    pydantic \
    httpx \
    requests \
    websockets \
    python-dotenv \
    rich

log "Python dependencies installed"

# ═══════════════════════════════════════════════════════════════════════════
# Step 6: Configure WOLF_AI for Termux
# ═══════════════════════════════════════════════════════════════════════════
info "Step 6/6: Configuring WOLF_AI for Termux..."

# Create .env if not exists
if [ ! -f "$WOLF_DIR/.env" ]; then
    cp "$WOLF_DIR/.env.example" "$WOLF_DIR/.env"

    # Generate API key
    API_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")

    # Update .env for Termux
    sed -i "s|WOLF_ROOT=.*|WOLF_ROOT=$WOLF_DIR|g" "$WOLF_DIR/.env"
    sed -i "s|WOLF_API_KEY=.*|WOLF_API_KEY=$API_KEY|g" "$WOLF_DIR/.env"
    sed -i "s|WOLF_API_HOST=.*|WOLF_API_HOST=0.0.0.0|g" "$WOLF_DIR/.env"

    log "Generated API Key: $API_KEY"
    warn "Save this key! You'll need it to connect."
fi

# Create Termux boot script
mkdir -p ~/.termux/boot
cat > ~/.termux/boot/wolf_ai.sh << 'BOOTSCRIPT'
#!/data/data/com.termux/files/usr/bin/bash
# Start WOLF_AI on boot
termux-wake-lock
cd ~/WOLF_AI && python run_server.py &
BOOTSCRIPT
chmod +x ~/.termux/boot/wolf_ai.sh

# Create convenient aliases
cat >> ~/.bashrc << 'ALIASES'

# WOLF_AI Aliases
alias wolf='cd ~/WOLF_AI'
alias wolf-start='cd ~/WOLF_AI && python run_server.py'
alias wolf-status='curl -s -H "X-API-Key: $(grep WOLF_API_KEY ~/WOLF_AI/.env | cut -d= -f2)" http://localhost:8000/api/status | jq'
alias wolf-awaken='curl -s -X POST -H "X-API-Key: $(grep WOLF_API_KEY ~/WOLF_AI/.env | cut -d= -f2)" http://localhost:8000/api/awaken | jq'
alias wolf-howl='f(){ curl -s -X POST -H "Content-Type: application/json" -H "X-API-Key: $(grep WOLF_API_KEY ~/WOLF_AI/.env | cut -d= -f2)" -d "{\"message\":\"$1\"}" http://localhost:8000/api/howl | jq; }; f'

ALIASES

log "Termux configuration complete"

# ═══════════════════════════════════════════════════════════════════════════
# Done!
# ═══════════════════════════════════════════════════════════════════════════

echo "
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║   🐺 WOLF_AI Setup Complete!                                              ║
║                                                                           ║
║   Your Samsung S24 Ultra is now THE FOREST                                ║
║                                                                           ║
║   Commands:                                                               ║
║   ─────────────────────────────────────────────────────                   ║
║   wolf-start     - Start WOLF_AI server                                   ║
║   wolf-status    - Check pack status                                      ║
║   wolf-awaken    - Awaken the pack                                        ║
║   wolf-howl \"msg\" - Send howl to pack                                    ║
║                                                                           ║
║   Quick Start:                                                            ║
║   ─────────────────────────────────────────────────────                   ║
║   1. source ~/.bashrc                                                     ║
║   2. wolf-start                                                           ║
║   3. Open browser: http://localhost:8000/dashboard                        ║
║                                                                           ║
║   Your API Key is saved in ~/WOLF_AI/.env                                 ║
║                                                                           ║
║   AUUUUUUUUUUUUUUUUUUUUUUUUU!                                              ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
"

# Reload bashrc
source ~/.bashrc 2>/dev/null || true

log "Run 'wolf-start' to begin!"
