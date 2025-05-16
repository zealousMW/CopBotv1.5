'use client';

import { useEffect } from 'react';
import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// Fix for default marker icons
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.3.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.3.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.3.1/images/marker-shadow.png',
});

// Sample police stations data - in a real app, this would come from an API
const SAMPLE_POLICE_STATIONS = [
  {
    position: [0, 0] as [number, number],
    name: "Central Police Station",
    distance: "1.2 km"
  },
  {
    position: [0, 0] as [number, number],
    name: "Downtown Police Post",
    distance: "2.5 km"
  }
];

function ChangeView({ center }: { center: [number, number] }) {
  const map = useMap();
  useEffect(() => {
    map.setView(center);
  }, [center, map]);
  return null;
}

type Props = {
  center: [number, number];
};

const Map = ({ center }: Props) => {
  // Generate sample police stations around the user's location
  const policeStations = SAMPLE_POLICE_STATIONS.map(station => ({
    ...station,
    position: [
      center[0] + (Math.random() - 0.5) * 0.02,
      center[1] + (Math.random() - 0.5) * 0.02
    ] as [number, number]
  }));

  return (
    <div className="h-[300px] w-full rounded-lg overflow-hidden shadow-lg">
      <MapContainer
        center={center}
        zoom={13}
        style={{ height: '100%', width: '100%' }}
      >
        <ChangeView center={center} />
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        />
        <Marker position={center}>
          <Popup>Your Location</Popup>
        </Marker>
        {policeStations.map((station, index) => (
          <Marker key={index} position={station.position}>
            <Popup>
              <div>
                <h3 className="font-bold">{station.name}</h3>
                {station.distance && <p>Distance: {station.distance}</p>}
              </div>
            </Popup>
          </Marker>
        ))}
      </MapContainer>
    </div>  );
};

export default Map;
