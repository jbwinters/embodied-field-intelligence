// EFI Research Page JavaScript

// Load and render charts when page loads
document.addEventListener('DOMContentLoaded', async () => {
    // These charts use the archived chemotaxis experiment, not pilot data.
    try {
        if (typeof Chart === 'undefined') throw new Error('Chart library unavailable');
        const response = await fetch('assets/data/experiment_summary.json');
        if (!response.ok) throw new Error(`Chart data unavailable (${response.status})`);
        const data = await response.json();
        document.querySelectorAll('.result-card canvas').forEach(el => { el.hidden = false; });
        renderAblationChart(data.ablation_results);
        renderScalingChart(data.scale_results);
        renderSensitivityChart(data.sensitivity_results);
        document.querySelectorAll('[data-chart-fallback]').forEach(el => { el.hidden = true; });
    } catch (error) {
        // Keep the actual archived figures; never substitute invented measurements.
        document.querySelectorAll('.result-card canvas').forEach(el => { el.hidden = true; });
        const status = document.getElementById('chart-status');
        status.textContent = 'Interactive charts are unavailable. Showing archived figures; measurements remain in assets/data/experiment_summary.json.';
        status.hidden = false;
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
                data: data.map(d => d.mean),
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
                error: data.map(d => d.std)
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
            labels: data.map(d => `${d.size}×${d.size}`),
            datasets: [{
                label: 'Mean Return',
                data: data.map(d => d.mean),
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
            labels: data.map(d => d.diff.toFixed(2)),
            datasets: [{
                label: 'Mean Return',
                data: data.map(d => d.mean),
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