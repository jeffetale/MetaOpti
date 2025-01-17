// src/app/page.js

import TradingControls from './components/TradingControls';

export default function Home() {
  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">Trading Bot Dashboard</h1>
      <TradingControls />
    </div>
  );
}