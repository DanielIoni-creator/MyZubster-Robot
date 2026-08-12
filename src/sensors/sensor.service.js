/**
 * 📡 Sensor Service - Gestione Sensori Robot
 */

class SensorService {
    constructor() {
        this.sensors = {
            temperature: { value: 22.5, unit: '°C', status: 'normal' },
            humidity: { value: 65, unit: '%', status: 'normal' },
            soilMoisture: { value: 45, unit: '%', status: 'normal' },
            light: { value: 800, unit: 'lux', status: 'normal' },
            battery: { value: 85, unit: '%', status: 'normal' },
            distance: { value: 0, unit: 'cm', status: 'normal' }
        };
        this.history = [];
        this.alerts = [];
        this.initSensors();
    }

    // Inizializza sensori
    initSensors() {
        // Genera dati iniziali
        for (let i = 0; i < 10; i++) {
            this.generateReading();
        }
        this.startMonitoring();
    }

    // Genera lettura
    generateReading() {
        const reading = {
            timestamp: new Date().toISOString(),
            temperature: 18 + Math.random() * 8,
            humidity: 50 + Math.random() * 30,
            soilMoisture: 30 + Math.random() * 40,
            light: 300 + Math.random() * 500,
            battery: 80 + Math.random() * 15,
            distance: Math.random() * 50
        };
        
        this.history.push(reading);
        if (this.history.length > 100) {
            this.history.shift();
        }

        // Aggiorna valori attuali
        this.sensors.temperature.value = reading.temperature;
        this.sensors.humidity.value = reading.humidity;
        this.sensors.soilMoisture.value = reading.soilMoisture;
        this.sensors.light.value = reading.light;
        this.sensors.battery.value = reading.battery;
        this.sensors.distance.value = reading.distance;

        // Controlla allarmi
        this.checkAlerts(reading);
    }

    // Avvia monitoraggio
    startMonitoring() {
        setInterval(() => {
            this.generateReading();
        }, 5000); // Ogni 5 secondi
    }

    // Controlla allarmi
    checkAlerts(reading) {
        if (reading.temperature > 30) {
            this.addAlert('temperature', 'Temperatura elevata', reading.temperature);
        }
        if (reading.soilMoisture < 20) {
            this.addAlert('soilMoisture', 'Umidità terreno bassa', reading.soilMoisture);
        }
        if (reading.battery < 20) {
            this.addAlert('battery', 'Batteria scarica', reading.battery);
        }
        if (reading.light < 100) {
            this.addAlert('light', 'Poca luce', reading.light);
        }
    }

    // Aggiungi allarme
    addAlert(type, message, value) {
        const alert = {
            id: `alert_${Date.now()}`,
            type: type,
            message: message,
            value: value,
            timestamp: new Date().toISOString(),
            status: 'active'
        };
        this.alerts.push(alert);
    }

    // Ottieni dati sensori
    async getSensorData() {
        try {
            return {
                success: true,
                current: this.sensors,
                history: this.history.slice(-20),
                alerts: this.alerts.filter(a => a.status === 'active'),
                timestamp: new Date().toISOString()
            };
        } catch (error) {
            console.error('❌ Errore getSensorData:', error);
            return { success: false, error: error.message };
        }
    }

    // Ottieni allarmi
    async getAlerts() {
        try {
            return {
                success: true,
                alerts: this.alerts.filter(a => a.status === 'active'),
                total: this.alerts.length
            };
        } catch (error) {
            console.error('❌ Errore getAlerts:', error);
            return { success: false, error: error.message };
        }
    }

    // Risolvi allarme
    async resolveAlert(alertId) {
        try {
            const alert = this.alerts.find(a => a.id === alertId);
            if (!alert) {
                throw new Error('Allarme non trovato');
            }
            alert.status = 'resolved';
            alert.resolvedAt = new Date().toISOString();
            return {
                success: true,
                message: '✅ Allarme risolto',
                alert: alert
            };
        } catch (error) {
            console.error('❌ Errore resolveAlert:', error);
            return { success: false, error: error.message };
        }
    }

    // Ottieni statistiche sensori
    async getStats() {
        try {
            const history = this.history.slice(-20);
            if (history.length === 0) {
                return { success: true, stats: {} };
            }

            const stats = {
                temperature: {
                    min: Math.min(...history.map(h => h.temperature)),
                    max: Math.max(...history.map(h => h.temperature)),
                    avg: history.reduce((sum, h) => sum + h.temperature, 0) / history.length
                },
                humidity: {
                    min: Math.min(...history.map(h => h.humidity)),
                    max: Math.max(...history.map(h => h.humidity)),
                    avg: history.reduce((sum, h) => sum + h.humidity, 0) / history.length
                },
                soilMoisture: {
                    min: Math.min(...history.map(h => h.soilMoisture)),
                    max: Math.max(...history.map(h => h.soilMoisture)),
                    avg: history.reduce((sum, h) => sum + h.soilMoisture, 0) / history.length
                },
                light: {
                    min: Math.min(...history.map(h => h.light)),
                    max: Math.max(...history.map(h => h.light)),
                    avg: history.reduce((sum, h) => sum + h.light, 0) / history.length
                }
            };

            return {
                success: true,
                stats: stats,
                period: 'ultime 20 letture'
            };
        } catch (error) {
            console.error('❌ Errore getStats:', error);
            return { success: false, error: error.message };
        }
    }

    // Calibra sensori
    async calibrate() {
        try {
            // Simula calibrazione
            await new Promise(resolve => setTimeout(resolve, 2000));
            return {
                success: true,
                message: '✅ Sensori calibrati con successo',
                timestamp: new Date().toISOString()
            };
        } catch (error) {
            console.error('❌ Errore calibrate:', error);
            return { success: false, error: error.message };
        }
    }
}

export const sensorService = new SensorService();
