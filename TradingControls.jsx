// TradingControls.jsx

import React, { useState, useEffect } from 'react';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Volume2, VolumeX, Ban, DollarSign, Trash2, Power } from 'lucide-react';

const TradingControls = () => {
  const [lastAction, setLastAction] = useState('');
  const [soundEnabled, setSoundEnabled] = useState(true);
  const [showAlert, setShowAlert] = useState(false);
  
  useEffect(() => {
    const handleKeyPress = (event) => {
      // Only trigger if Shift + Alt are pressed with the key
      if (event.shiftKey && event.altKey) {
        switch (event.key.toUpperCase()) {
          case 'T':
            setLastAction('Closing all profitable positions');
            setShowAlert(true);
            break;
          case 'L':
            setLastAction('Closing all losing positions');
            setShowAlert(true);
            break;
          case 'M':
            setSoundEnabled(!soundEnabled);
            setLastAction(`Sound alerts ${!soundEnabled ? 'enabled' : 'disabled'}`);
            setShowAlert(true);
            break;
          case 'N':
            setLastAction('New trade opening has been disabled');
            setShowAlert(true);
            break;
          default:
            return;
        }
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [soundEnabled]);

  // Hide alert after 3 seconds
  useEffect(() => {
    if (showAlert) {
      const timer = setTimeout(() => setShowAlert(false), 3000);
      return () => clearTimeout(timer);
    }
  }, [showAlert]);

  return (
    <div className="space-y-4 p-4">
      <Card>
        <CardHeader>
          <CardTitle className="text-lg font-bold">Trading Controls</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-4">
            <div className="flex items-center space-x-2">
              <DollarSign className="w-4 h-4" />
              <span>Shift + Alt + T: Close profitable positions</span>
            </div>
            <div className="flex items-center space-x-2">
              <Trash2 className="w-4 h-4" />
              <span>Shift + Alt + L: Close losing positions</span>
            </div>
            <div className="flex items-center space-x-2">
              {soundEnabled ? (
                <Volume2 className="w-4 h-4" />
              ) : (
                <VolumeX className="w-4 h-4" />
              )}
              <span>Shift + Alt + M: Toggle sound alerts</span>
            </div>
            <div className="flex items-center space-x-2">
              <Ban className="w-4 h-4" />
              <span>Shift + Alt + N: Stop new trades</span>
            </div>
          </div>
        </CardContent>
      </Card>

      {showAlert && (
        <Alert variant="default" className="mt-4">
          <AlertTitle>Action Triggered</AlertTitle>
          <AlertDescription>{lastAction}</AlertDescription>
        </Alert>
      )}

      <div className="text-sm text-gray-500 mt-4">
        Note: All controls require Shift + Alt + Key combination to prevent accidental activation
      </div>
    </div>
  );
};

export default TradingControls;
