/**
 * Chart configurations for the dashboard
 * This file contains standard configurations for Chart.js charts
 */

// Common chart options
const chartDefaults = {
    // Animation settings
    animation: {
        duration: 1000,
        easing: 'easeOutQuart'
    },
    // Responsive design
    responsive: true,
    maintainAspectRatio: false,
    // Layout
    layout: {
        padding: {
            left: 10,
            right: 25,
            top: 25,
            bottom: 0
        }
    },
    // Interaction settings
    interaction: {
        intersect: false,
        mode: 'index'
    },
    // Plugins
    plugins: {
        legend: {
            display: true,
            position: 'top',
            labels: {
                font: {
                    size: 12,
                    family: "'Nunito', 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
                },
                padding: 20
            }
        },
        tooltip: {
            backgroundColor: 'rgba(0, 0, 0, 0.8)',
            titleFont: {
                size: 14,
                family: "'Nunito', 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif",
                weight: 'bold'
            },
            bodyFont: {
                family: "'Nunito', 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
            },
            borderWidth: 0,
            cornerRadius: 4,
            caretSize: 6,
            xPadding: 12,
            yPadding: 12
        }
    }
};

// Bar chart defaults
const barChartDefaults = {
    ...chartDefaults,
    scales: {
        x: {
            grid: {
                drawBorder: false,
                display: false
            },
            ticks: {
                padding: 10
            }
        },
        y: {
            grid: {
                drawBorder: false,
                display: true,
                drawTicks: false,
                borderDash: [5, 5]
            },
            ticks: {
                padding: 10,
                beginAtZero: true
            }
        }
    }
};

// Line chart defaults
const lineChartDefaults = {
    ...chartDefaults,
    elements: {
        line: {
            tension: 0.3,
            borderWidth: 2
        },
        point: {
            radius: 4,
            hitRadius: 10,
            hoverRadius: 6,
            hoverBorderWidth: 2
        }
    },
    scales: {
        x: {
            grid: {
                display: false
            }
        },
        y: {
            grid: {
                borderDash: [5, 5],
                drawBorder: false
            },
            ticks: {
                beginAtZero: true
            }
        }
    }
};

// Doughnut/Pie chart defaults
const doughnutChartDefaults = {
    ...chartDefaults,
    cutout: '70%',
    plugins: {
        ...chartDefaults.plugins,
        legend: {
            ...chartDefaults.plugins.legend,
            position: 'bottom'
        }
    }
};

// Chart color palettes
const chartColors = {
    primary: [
        'rgba(78, 115, 223, 0.8)',
        'rgba(54, 185, 204, 0.8)',
        'rgba(246, 194, 62, 0.8)',
        'rgba(231, 74, 59, 0.8)',
        'rgba(90, 92, 105, 0.8)'
    ],
    secondary: [
        'rgba(66, 139, 202, 0.8)',
        'rgba(92, 184, 92, 0.8)',
        'rgba(240, 173, 78, 0.8)',
        'rgba(217, 83, 79, 0.8)',
        'rgba(150, 151, 162, 0.8)'
    ]
};

// Function to initialize a bar chart
function initBarChart(ctx, labels, data, options = {}) {
    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: chartColors.primary,
                borderColor: chartColors.primary.map(color => color.replace('0.8', '1')),
                borderWidth: 1
            }]
        },
        options: {
            ...barChartDefaults,
            ...options
        }
    });
}

// Function to initialize a line chart
function initLineChart(ctx, labels, data, options = {}) {
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: 'rgba(78, 115, 223, 0.05)',
                borderColor: 'rgba(78, 115, 223, 1)',
                pointBackgroundColor: 'rgba(78, 115, 223, 1)',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: 'rgba(78, 115, 223, 1)',
                fill: true
            }]
        },
        options: {
            ...lineChartDefaults,
            ...options
        }
    });
}

// Function to initialize a doughnut chart
function initDoughnutChart(ctx, labels, data, options = {}) {
    return new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: chartColors.primary,
                borderColor: '#ffffff',
                borderWidth: 2
            }]
        },
        options: {
            ...doughnutChartDefaults,
            ...options
        }
    });
}

// Function to update an existing chart
function updateChart(chart, labels, data) {
    chart.data.labels = labels;
    chart.data.datasets[0].data = data;
    chart.update();
}
