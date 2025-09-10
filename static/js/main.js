// Main JavaScript for Breed Recognition System

// Global configuration
const CONFIG = {
    MAX_FILE_SIZE: 16 * 1024 * 1024, // 16MB
    ALLOWED_TYPES: ['image/png', 'image/jpg', 'image/jpeg', 'image/gif'],
    UPLOAD_ENDPOINT: '/predict',
    API_ENDPOINTS: {
        MODEL_INFO: '/api/model_info',
        BREEDS: '/api/breeds',
        HEALTH: '/health'
    }
};

// Utility functions
const Utils = {
    /**
     * Format file size in human readable format
     */
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    },

    /**
     * Validate file type and size
     */
    validateFile(file) {
        const errors = [];

        // Check file type
        if (!CONFIG.ALLOWED_TYPES.includes(file.type)) {
            errors.push('Invalid file type. Please select a PNG, JPG, JPEG, or GIF image.');
        }

        // Check file size
        if (file.size > CONFIG.MAX_FILE_SIZE) {
            errors.push(`File size too large. Maximum size is ${this.formatFileSize(CONFIG.MAX_FILE_SIZE)}.`);
        }

        // Check if file is actually an image
        if (!file.type.startsWith('image/')) {
            errors.push('Selected file is not an image.');
        }

        return {
            isValid: errors.length === 0,
            errors: errors
        };
    },

    /**
     * Show notification
     */
    showNotification(message, type = 'info', duration = 5000) {
        const notification = document.createElement('div');
        notification.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
        notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
        
        notification.innerHTML = `
            <div class="d-flex align-items-center">
                <i class="fas fa-${this.getNotificationIcon(type)} me-2"></i>
                <span>${message}</span>
                <button type="button" class="btn-close ms-auto" data-bs-dismiss="alert"></button>
            </div>
        `;

        document.body.appendChild(notification);

        // Auto remove after duration
        setTimeout(() => {
            if (notification && notification.parentNode) {
                notification.remove();
            }
        }, duration);
    },

    /**
     * Get notification icon based on type
     */
    getNotificationIcon(type) {
        const icons = {
            'success': 'check-circle',
            'danger': 'exclamation-circle',
            'warning': 'exclamation-triangle',
            'info': 'info-circle'
        };
        return icons[type] || 'info-circle';
    },

    /**
     * Create loading spinner
     */
    createLoadingSpinner(text = 'Loading...') {
        return `
            <div class="text-center">
                <div class="spinner-border text-primary mb-3" role="status">
                    <span class="visually-hidden">${text}</span>
                </div>
                <p class="text-muted">${text}</p>
            </div>
        `;
    },

    /**
     * Debounce function
     */
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },

    /**
     * Animate element
     */
    animateElement(element, animationClass = 'fade-in') {
        element.classList.add(animationClass);
        element.addEventListener('animationend', () => {
            element.classList.remove(animationClass);
        }, { once: true });
    }
};

// Image processing utilities
const ImageUtils = {
    /**
     * Create image preview
     */
    createPreview(file, callback) {
        if (!file || !file.type.startsWith('image/')) {
            callback(null);
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                // Get image dimensions and other metadata
                const metadata = {
                    width: img.width,
                    height: img.height,
                    aspectRatio: (img.width / img.height).toFixed(2),
                    fileSize: Utils.formatFileSize(file.size),
                    fileName: file.name,
                    fileType: file.type,
                    dataUrl: e.target.result
                };
                callback(metadata);
            };
            img.onerror = () => callback(null);
            img.src = e.target.result;
        };
        reader.onerror = () => callback(null);
        reader.readAsDataURL(file);
    },

    /**
     * Compress image if needed
     */
    compressImage(file, maxWidth = 1024, quality = 0.8) {
        return new Promise((resolve) => {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            const img = new Image();

            img.onload = () => {
                // Calculate new dimensions
                let { width, height } = img;
                if (width > maxWidth) {
                    height = (height * maxWidth) / width;
                    width = maxWidth;
                }

                canvas.width = width;
                canvas.height = height;

                // Draw and compress
                ctx.drawImage(img, 0, 0, width, height);
                canvas.toBlob(resolve, 'image/jpeg', quality);
            };

            img.onerror = () => resolve(file); // Return original if compression fails
            img.src = URL.createObjectURL(file);
        });
    }
};

// API utilities
const API = {
    /**
     * Make API request
     */
    async request(url, options = {}) {
        try {
            const response = await fetch(url, {
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers
                },
                ...options
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('API request failed:', error);
            throw error;
        }
    },

    /**
     * Upload file for prediction
     */
    async predict(file, onProgress = null) {
        const formData = new FormData();
        formData.append('file', file);

        return new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();

            // Progress tracking
            if (onProgress) {
                xhr.upload.addEventListener('progress', (e) => {
                    if (e.lengthComputable) {
                        const percentComplete = (e.loaded / e.total) * 100;
                        onProgress(percentComplete);
                    }
                });
            }

            xhr.addEventListener('load', () => {
                if (xhr.status === 200) {
                    try {
                        const response = JSON.parse(xhr.responseText);
                        resolve(response);
                    } catch (error) {
                        reject(new Error('Invalid response format'));
                    }
                } else {
                    reject(new Error(`Upload failed with status: ${xhr.status}`));
                }
            });

            xhr.addEventListener('error', () => {
                reject(new Error('Network error occurred'));
            });

            xhr.open('POST', CONFIG.UPLOAD_ENDPOINT);
            xhr.send(formData);
        });
    },

    /**
     * Get model information
     */
    async getModelInfo() {
        return this.request(CONFIG.API_ENDPOINTS.MODEL_INFO);
    },

    /**
     * Get all breeds information
     */
    async getBreeds(language = 'en') {
        return this.request(`${CONFIG.API_ENDPOINTS.BREEDS}?language=${language}`);
    },

    /**
     * Health check
     */
    async healthCheck() {
        return this.request(CONFIG.API_ENDPOINTS.HEALTH);
    }
};

// Results formatter
const ResultsFormatter = {
    /**
     * Format prediction results for display
     */
    formatResults(data, translations) {
        const { is_single_prediction, predictions, confidence_message } = data;
        
        let html = '';

        // Title
        html += `<h4 class="mb-4 text-center">`;
        html += `<i class="fas fa-${is_single_prediction ? 'check-circle text-success' : 'list text-primary'} me-2"></i>`;
        html += is_single_prediction ? translations.prediction_single : translations.prediction_multiple;
        html += `</h4>`;

        // Confidence message
        if (confidence_message) {
            html += `<div class="alert alert-info mb-4">`;
            html += `<i class="fas fa-info-circle me-2"></i>${confidence_message}`;
            html += `</div>`;
        }

        // Process each prediction
        predictions.forEach((item, index) => {
            html += this.formatPredictionItem(item, index, translations);
        });

        return html;
    },

    /**
     * Format individual prediction item
     */
    formatPredictionItem(item, index, translations) {
        const { prediction, info } = item;
        const isTop = index === 0;

        let html = `<div class="breed-result mb-4 p-4 border rounded-3 ${isTop ? 'border-success bg-light' : 'border-secondary'}">`;

        // Header with breed name and confidence
        html += this.formatPredictionHeader(prediction, translations);

        // Breed information
        if (info) {
            html += this.formatBreedInfo(info, translations);
        }

        html += `</div>`;
        return html;
    },

    /**
     * Format prediction header
     */
    formatPredictionHeader(prediction, translations) {
        const confidenceClass = this.getConfidenceClass(prediction.confidence);
        const confidencePercentage = (prediction.confidence * 100).toFixed(1);

        return `
            <div class="d-flex justify-content-between align-items-center mb-3">
                <h5 class="mb-0 text-success fw-bold">${prediction.breed}</h5>
                <span class="badge bg-${confidenceClass} fs-6 px-3 py-2">
                    <i class="fas fa-chart-line me-1"></i>
                    ${translations.confidence}: ${confidencePercentage}%
                </span>
            </div>
        `;
    },

    /**
     * Format breed information
     */
    formatBreedInfo(info, translations) {
        let html = `<div class="row g-3">`;

        // Basic Information Column
        html += `<div class="col-md-6">`;
        html += `<h6 class="text-primary mb-3"><i class="fas fa-info-circle me-2"></i>${translations.breed_information}</h6>`;
        html += `<div class="list-group list-group-flush">`;
        
        if (info.origin) {
            html += `<div class="list-group-item border-0 px-0 py-2">
                <strong>${translations.origin}:</strong> <span class="text-muted">${info.origin}</span>
            </div>`;
        }
        if (info.type) {
            html += `<div class="list-group-item border-0 px-0 py-2">
                <strong>Type:</strong> <span class="text-muted">${info.type}</span>
            </div>`;
        }
        if (info.color) {
            html += `<div class="list-group-item border-0 px-0 py-2">
                <strong>${translations.color}:</strong> <span class="text-muted">${info.color}</span>
            </div>`;
        }
        
        html += `</div></div>`;

        // Characteristics Column
        html += `<div class="col-md-6">`;
        html += `<h6 class="text-primary mb-3"><i class="fas fa-chart-bar me-2"></i>${translations.characteristics}</h6>`;
        html += `<div class="list-group list-group-flush">`;

        if (info.milk_yield) {
            html += `<div class="list-group-item border-0 px-0 py-2">
                <strong>${translations.avg_milk_yield}:</strong> 
                <span class="text-muted">${info.milk_yield} ${translations.liters} ${translations.per_day}</span>
            </div>`;
        }

        if (info.body_weight) {
            if (info.body_weight.male) {
                html += `<div class="list-group-item border-0 px-0 py-2">
                    <strong>${translations.body_weight} (${translations.male}):</strong> 
                    <span class="text-muted">${info.body_weight.male} ${translations.kg}</span>
                </div>`;
            }
            if (info.body_weight.female) {
                html += `<div class="list-group-item border-0 px-0 py-2">
                    <strong>${translations.body_weight} (${translations.female}):</strong> 
                    <span class="text-muted">${info.body_weight.female} ${translations.kg}</span>
                </div>`;
            }
        }

        html += `</div></div>`;
        html += `</div>`;

        // Additional characteristics
        if (info.characteristics && info.characteristics.length > 0) {
            html += `<div class="mt-4">`;
            html += `<h6 class="text-primary mb-3"><i class="fas fa-star me-2"></i>Key Features</h6>`;
            html += `<div class="d-flex flex-wrap gap-2">`;
            info.characteristics.forEach(char => {
                html += `<span class="badge bg-secondary px-3 py-2">${char}</span>`;
            });
            html += `</div></div>`;
        }

        return html;
    },

    /**
     * Get confidence badge class
     */
    getConfidenceClass(confidence) {
        if (confidence >= 0.9) return 'success';
        if (confidence >= 0.7) return 'primary';
        if (confidence >= 0.5) return 'warning';
        return 'secondary';
    }
};

// Initialize application
const App = {
    /**
     * Initialize the application
     */
    init() {
        this.bindEvents();
        this.checkHealth();
    },

    /**
     * Bind event listeners
     */
    bindEvents() {
        // Handle window resize
        window.addEventListener('resize', Utils.debounce(this.handleResize.bind(this), 250));

        // Handle visibility change
        document.addEventListener('visibilitychange', this.handleVisibilityChange.bind(this));

        // Handle online/offline status
        window.addEventListener('online', this.handleOnline.bind(this));
        window.addEventListener('offline', this.handleOffline.bind(this));
    },

    /**
     * Handle window resize
     */
    handleResize() {
        // Adjust layout for mobile
        const isMobile = window.innerWidth < 768;
        document.body.classList.toggle('mobile-layout', isMobile);
    },

    /**
     * Handle visibility change
     */
    handleVisibilityChange() {
        if (document.visibilityState === 'visible') {
            // Check health when tab becomes visible
            this.checkHealth();
        }
    },

    /**
     * Handle online status
     */
    handleOnline() {
        Utils.showNotification('Connection restored', 'success');
        this.checkHealth();
    },

    /**
     * Handle offline status
     */
    handleOffline() {
        Utils.showNotification('Connection lost. Some features may not work.', 'warning');
    },

    /**
     * Check application health
     */
    async checkHealth() {
        try {
            const health = await API.healthCheck();
            if (!health.model_loaded) {
                Utils.showNotification('Model not loaded. Please contact administrator.', 'warning');
            }
        } catch (error) {
            console.warn('Health check failed:', error);
        }
    }
};

// Export for global use
window.BreedRecognition = {
    Utils,
    ImageUtils,
    API,
    ResultsFormatter,
    App,
    CONFIG
};

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    App.init();
});