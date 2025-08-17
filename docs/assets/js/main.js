// EFI Research Page JavaScript

// Load and render charts when page loads
document.addEventListener('DOMContentLoaded', async () => {
    // Load analysis data - try new comprehensive results first
    try {
        const response = await fetch('assets/data/experiment_summary.json');
        const data = await response.json();
        
        renderAblationChart(data.ablation_results);
        renderScalingChart(data.scale_results);
        renderSensitivityChart(data.sensitivity_results);
    } catch (error) {
        console.error('Error loading experiment data, trying old format:', error);
        try {
            const response = await fetch('assets/data/analysis_results.json');
            const data = await response.json();
            
            renderAblationChart(data.ablation);
            renderScalingChart(data.scaling);
            renderSensitivityChart(data.sensitivity);
        } catch (error2) {
            console.error('Error loading analysis data:', error2);
            // Use fallback data if file doesn't exist
            useFallbackData();
        }
    }
    
    // Smooth scrolling for navigation links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });
    
    // Add scroll animations
    observeElements();
});

function renderAblationChart(data) {
    const ctx = document.getElementById('ablationChart');
    if (!ctx) return;
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.map(d => d.condition),
            datasets: [{
                label: 'Mean Return',
                data: data.map(d => d.mean_return),
                backgroundColor: [
                    'rgba(76, 175, 80, 0.7)',
                    'rgba(255, 152, 0, 0.7)',
                    'rgba(255, 152, 0, 0.7)',
                    'rgba(255, 152, 0, 0.7)',
                    'rgba(244, 67, 54, 0.7)'
                ],
                borderColor: [
                    'rgb(76, 175, 80)',
                    'rgb(255, 152, 0)',
                    'rgb(255, 152, 0)',
                    'rgb(255, 152, 0)',
                    'rgb(244, 67, 54)'
                ],
                borderWidth: 2,
                error: data.map(d => d.std_return)
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: false
                },
                title: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    title: {
                        display: true,
                        text: 'Mean Return'
                    }
                },
                x: {
                    ticks: {
                        autoSkip: false,
                        maxRotation: 45,
                        minRotation: 45
                    }
                }
            }
        }
    });
}

function renderScalingChart(data) {
    const ctx = document.getElementById('scalingChart');
    if (!ctx) return;
    
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.map(d => `${d.grid_size}×${d.grid_size}`),
            datasets: [{
                label: 'Mean Return',
                data: data.map(d => d.mean_return),
                borderColor: 'rgb(33, 150, 243)',
                backgroundColor: 'rgba(33, 150, 243, 0.1)',
                tension: 0.3,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    title: {
                        display: true,
                        text: 'Mean Return'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Grid Size'
                    }
                }
            }
        }
    });
}

function renderSensitivityChart(data) {
    const ctx = document.getElementById('sensitivityChart');
    if (!ctx) return;
    
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.map(d => d.diffusion_rate.toFixed(2)),
            datasets: [{
                label: 'Mean Return',
                data: data.map(d => d.mean_return),
                borderColor: 'rgb(156, 39, 176)',
                backgroundColor: 'rgba(156, 39, 176, 0.1)',
                tension: 0.3,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: false
                },
                annotation: {
                    annotations: {
                        line1: {
                            type: 'line',
                            xMin: '0.25',
                            xMax: '0.25',
                            borderColor: 'rgb(255, 99, 132)',
                            borderWidth: 2,
                            borderDash: [5, 5],
                            label: {
                                content: 'Default',
                                enabled: true,
                                position: 'start'
                            }
                        }
                    }
                }
            },
            scales: {
                y: {
                    title: {
                        display: true,
                        text: 'Mean Return'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Diffusion Rate'
                    }
                }
            }
        }
    });
}

function useFallbackData() {
    // Fallback data if JSON file is not available
    const fallbackAblation = [
        { condition: 'Full Model', mean_return: -0.92, std_return: 0.95 },
        { condition: 'No Trail', mean_return: -2.0, std_return: 0.0 },
        { condition: 'No Novelty', mean_return: -0.88, std_return: 0.95 },
        { condition: 'No Corner', mean_return: -1.12, std_return: 0.84 },
        { condition: 'Baseline Only', mean_return: -1.8, std_return: 0.4 }
    ];
    
    const fallbackScaling = [
        { grid_size: 10, mean_return: -1.68 },
        { grid_size: 15, mean_return: -0.87 },
        { grid_size: 20, mean_return: -0.13 },
        { grid_size: 25, mean_return: -0.87 },
        { grid_size: 30, mean_return: 0.8 }
    ];
    
    const fallbackSensitivity = [
        { diffusion_rate: 0.05, mean_return: -0.87 },
        { diffusion_rate: 0.10, mean_return: -1.13 },
        { diffusion_rate: 0.15, mean_return: -1.2 },
        { diffusion_rate: 0.20, mean_return: -1.13 },
        { diffusion_rate: 0.25, mean_return: -0.87 },
        { diffusion_rate: 0.30, mean_return: -1.13 }
    ];
    
    renderAblationChart(fallbackAblation);
    renderScalingChart(fallbackScaling);
    renderSensitivityChart(fallbackSensitivity);
}

function observeElements() {
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -100px 0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);
    
    // Observe all cards and sections
    document.querySelectorAll('.approach-card, .result-card, .demo-card, .future-card, .stat-box')
        .forEach(el => observer.observe(el));
}

// Add fade-in animation styles
const style = document.createElement('style');
style.textContent = `
    .fade-in {
        animation: fadeIn 0.6s ease-in-out;
    }
    
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
`;
document.head.appendChild(style);