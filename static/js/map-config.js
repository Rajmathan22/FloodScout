/**
 * Map configuration for the flood risk visualization
 * Uses Leaflet.js for interactive maps
 */

// Default map settings
const defaultMapSettings = {
    center: [20.5937, 78.9629], // Default to center of India
    zoom: 5,
    minZoom: 3,
    maxZoom: 18
};

// Different basemap tile layers
const basemaps = {
    OpenStreetMap: L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    }),
    Satellite: L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
        attribution: 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
    }),
    Terrain: L.tileLayer('https://stamen-tiles-{s}.a.ssl.fastly.net/terrain/{z}/{x}/{y}{r}.png', {
        attribution: 'Map tiles by <a href="http://stamen.com">Stamen Design</a>, <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a> &mdash; Map data &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    })
};

// Icon options
const markerOptions = {
    standard: {
        icon: L.icon({
            iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-blue.png',
            shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
            iconSize: [25, 41],
            iconAnchor: [12, 41],
            popupAnchor: [1, -34],
            shadowSize: [41, 41]
        })
    },
    risk: {
        icon: L.icon({
            iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-red.png',
            shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
            iconSize: [25, 41],
            iconAnchor: [12, 41],
            popupAnchor: [1, -34],
            shadowSize: [41, 41]
        })
    }
};

// Heatmap gradient for flood risk
const heatmapGradient = {
    0.4: 'blue',
    0.65: 'yellow',
    0.9: 'red'
};

/**
 * Initialize a Leaflet map with the given options
 * @param {string} containerId - ID of the container element
 * @param {Object} options - Map options
 * @return {Object} - The created map object
 */
function initMap(containerId, options = {}) {
    // Merge default settings with provided options
    const mapSettings = { ...defaultMapSettings, ...options };
    
    // Create map
    const map = L.map(containerId, {
        center: mapSettings.center,
        zoom: mapSettings.zoom,
        minZoom: mapSettings.minZoom,
        maxZoom: mapSettings.maxZoom
    });
    
    // Add default basemap
    basemaps.OpenStreetMap.addTo(map);
    
    // Add layer control if multiple basemaps are requested
    if (mapSettings.showLayerControl) {
        L.control.layers(basemaps).addTo(map);
    }
    
    // Add scale control
    if (mapSettings.showScaleControl) {
        L.control.scale().addTo(map);
    }
    
    return map;
}

/**
 * Add markers to the map for flood risk points
 * @param {Object} map - Leaflet map object
 * @param {Array} data - Array of data points with lat, lng, and risk properties
 * @param {boolean} cluster - Whether to cluster markers
 */
function addFloodRiskMarkers(map, data, cluster = true) {
    // Create a marker cluster group if clustering is enabled
    let markers = cluster ? L.markerClusterGroup() : L.layerGroup();
    
    // Add markers for each data point
    data.forEach(point => {
        const marker = L.marker([point.lat, point.lng], {
            icon: point.risk > 0.5 ? markerOptions.risk.icon : markerOptions.standard.icon
        });
        
        // Add popup with information
        const popupContent = `
            <div class="map-popup">
                <h6>Location: ${point.lat.toFixed(4)}, ${point.lng.toFixed(4)}</h6>
                <p><strong>Flood Risk Probability:</strong> ${(point.risk * 100).toFixed(2)}%</p>
                ${point.additionalInfo ? `<p>${point.additionalInfo}</p>` : ''}
            </div>
        `;
        marker.bindPopup(popupContent);
        
        // Add to marker group
        markers.addLayer(marker);
    });
    
    // Add markers to map
    map.addLayer(markers);
    
    return markers;
}

/**
 * Add a heatmap layer to the map for flood risk visualization
 * @param {Object} map - Leaflet map object
 * @param {Array} data - Array of data points with lat, lng, and intensity properties
 * @param {Object} options - Heatmap options
 */
function addHeatmap(map, data, options = {}) {
    // Format data for heatmap
    const heatData = data.map(point => [
        point.lat,
        point.lng,
        point.intensity || point.risk || 0.5
    ]);
    
    // Create heatmap layer
    const heat = L.heatLayer(heatData, {
        radius: options.radius || 15,
        blur: options.blur || 15,
        maxZoom: options.maxZoom || 17,
        max: options.max || 1.0,
        gradient: options.gradient || heatmapGradient
    });
    
    // Add to map
    map.addLayer(heat);
    
    return heat;
}
