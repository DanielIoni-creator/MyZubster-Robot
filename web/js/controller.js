// Stato robot
let robotState = {
    online: true,
    battery: 85,
    position: { x: 0, y: 0 },
    task: 'idle'
};

// Joystick
const joystick = document.getElementById('joystick');
const thumb = document.getElementById('joystickThumb');
let isDragging = false;

// Movimento joystick
thumb.addEventListener('mousedown', (e) => {
    isDragging = true;
    document.addEventListener('mousemove', moveJoystick);
    document.addEventListener('mouseup', stopJoystick);
});

function moveJoystick(e) {
    if (!isDragging) return;
    
    const rect = joystick.getBoundingClientRect();
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;
    
    const deltaX = e.clientX - centerX;
    const deltaY = e.clientY - centerY;
    
    const maxDist = rect.width / 2 - 25;
    const dist = Math.sqrt(deltaX * deltaX + deltaY * deltaY);
    const angle = Math.atan2(deltaY, deltaX);
    
    const moveX = Math.min(dist, maxDist) * Math.cos(angle);
    const moveY = Math.min(dist, maxDist) * Math.sin(angle);
    
    thumb.style.transform = `translate(calc(-50% + ${moveX}px), calc(-50% + ${moveY}px))`;
}

function stopJoystick() {
    isDragging = false;
    thumb.style.transform = 'translate(-50%, -50%)';
    document.removeEventListener('mousemove', moveJoystick);
    document.removeEventListener('mouseup', stopJoystick);
}

// Pulsanti direzione
document.querySelectorAll('.dir-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const direction = btn.dataset.dir;
        moveRobot(direction);
    });
});

function moveRobot(direction) {
    console.log(`🤖 Movimento: ${direction}`);
    // Simula movimento
    switch(direction) {
        case 'north': robotState.position.y += 1; break;
        case 'south': robotState.position.y -= 1; break;
        case 'east': robotState.position.x += 1; break;
        case 'west': robotState.position.x -= 1; break;
    }
    updateStatus();
}

// Missioni
function startMission() {
    const mission = document.getElementById('missionSelect').value;
    const status = document.getElementById('missionStatus');
    
    status.innerHTML = `🚀 Missione "${mission}" avviata...`;
    
    setTimeout(() => {
        status.innerHTML = `✅ Missione "${mission}" completata!`;
        robotState.task = 'idle';
    }, 3000);
    
    robotState.task = mission;
}

// Camera
function captureImage() {
    const img = document.getElementById('cameraFeed');
    // Simula cattura
    img.src = 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><text y=".9em" font-size="90">📷</text></svg>';
}

// Aggiorna stato
function updateStatus() {
    document.getElementById('batteryLevel').textContent = `${robotState.battery}%`;
}

// Inizializza
setInterval(() => {
    // Simula aggiornamenti sensori
    document.getElementById('temperature').textContent = `${(18 + Math.random() * 8).toFixed(1)}°C`;
    document.getElementById('humidity').textContent = `${(50 + Math.random() * 30).toFixed(0)}%`;
    document.getElementById('light').textContent = `${Math.floor(300 + Math.random() * 500)} lux`;
}, 2000);

console.log('🤖 Robot Controller inizializzato!');
