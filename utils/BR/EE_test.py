#!/usr/bin/env python3
"""
MQTT End-Effector Test Script
Streams variables to TOS/ee topic at 100Hz with specified patterns:
- Grind (M101): toggles between 1/0 every 5 seconds
- Setpoint (MW 0100): ramps 0->4095->0 every 10 seconds
"""

import paho.mqtt.client as mqtt
import json
import time
import math
import signal
import sys
from threading import Event

class EETestStreamer:
    def __init__(self):
        # MQTT Configuration
        self.broker_host = "localhost"  # You are the broker
        self.broker_port = 1883
        self.client_id = "EE_TestScript"  # Changed to avoid conflict with PLC
        self.topic = "TOS/ee"
        
        # Timing configuration
        self.stream_frequency = 10  # Hz
        self.stream_interval = 1.0 / self.stream_frequency  # 0.01 seconds (10ms)
        
        # Pattern configuration
        self.grind_toggle_interval = 50.0  # seconds
        self.select_toggle_interval = 60.0  # seconds
        self.setpoint_cycle_interval = 60.0  # seconds
        
        # State variables
        self.start_time = None
        self.running = False
        self.stop_event = Event()
        
        # MQTT client
        self.client = None
        
        # Statistics
        self.message_count = 0
        self.last_stats_time = None
        
    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print(f"✓ Connected to MQTT broker {self.broker_host}:{self.broker_port}")
            print(f"✓ Client ID: {self.client_id}")
            print(f"✓ Publishing to topic: {self.topic}")
        else:
            print(f"✗ Failed to connect to MQTT broker. Return code: {rc}")
            
    def on_publish(self, client, userdata, mid):
        # Uncomment for debugging publish confirmations
        # print(f"Message {mid} published")
        pass
        
    def on_disconnect(self, client, userdata, rc):
        if rc != 0:
            print(f"⚠ Unexpected disconnection from MQTT broker. Return code: {rc}")
        else:
            print("✓ Disconnected from MQTT broker")
            
    def calculate_grind_value(self, elapsed_time):
        """Calculate grind value: toggles between True/False every 5 seconds"""
        cycle_position = elapsed_time % self.grind_toggle_interval
        return cycle_position < (self.grind_toggle_interval / 2)
        
    def calculate_select_value(self, elapsed_time):
        """Calculate select value: toggles between True/False every 10 seconds"""
        cycle_position = elapsed_time % self.select_toggle_interval
        return cycle_position < (self.select_toggle_interval / 2)
        
    def calculate_setpoint_value(self, elapsed_time):
        """Calculate setpoint value: ramps 0->4095->0 every 10 seconds"""
        cycle_position = elapsed_time % self.setpoint_cycle_interval
        normalized_position = cycle_position / self.setpoint_cycle_interval
        
        # Create triangle wave: 0 -> 1 -> 0
        if normalized_position <= 0.5:
            # Rising: 0 to 1
            triangle_value = normalized_position * 2
        else:
            # Falling: 1 to 0
            triangle_value = 2 - (normalized_position * 2)
            
        # Scale to 0-4095 range
        return int(triangle_value * 4095)
        
    def create_payload(self, elapsed_time):
        """Create the MQTT payload with current values"""
        grind_value = self.calculate_grind_value(elapsed_time)
        select_value = self.calculate_select_value(elapsed_time)
        setpoint_value = self.calculate_setpoint_value(elapsed_time)
        
        payload = {
            "Grind": grind_value,      # M101: Boolean (True/False)
            "Select": select_value,    # M103: Boolean (True/False)
            "Setpoint": setpoint_value # MW 0100: Word (0-4095)
        }
        
        return json.dumps(payload)
        
    def print_status(self, elapsed_time, payload_str):
        """Print current status every second"""
        if self.last_stats_time is None or elapsed_time - self.last_stats_time >= 1.0:
            payload_data = json.loads(payload_str)
            
            # Calculate actual frequency
            if self.last_stats_time is not None:
                messages_this_second = self.message_count - getattr(self, 'last_message_count', 0)
                actual_freq = messages_this_second / (elapsed_time - self.last_stats_time)
            else:
                actual_freq = 0
                
            print(f"Time: {elapsed_time:6.1f}s | "
                  f"Grind: {str(payload_data['Grind']):5s} | "
                  f"Select: {str(payload_data['Select']):5s} | "
                  f"Setpoint: {payload_data['Setpoint']:4d} | "
                  f"Freq: {actual_freq:5.1f}Hz | "
                  f"Messages: {self.message_count}")
                  
            self.last_stats_time = elapsed_time
            self.last_message_count = self.message_count
            
    def setup_mqtt(self):
        """Initialize and configure MQTT client"""
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, client_id=self.client_id, protocol=mqtt.MQTTv311)
        
        # Set callbacks
        self.client.on_connect = self.on_connect
        self.client.on_publish = self.on_publish
        self.client.on_disconnect = self.on_disconnect
        
        try:
            print(f"Connecting to MQTT broker {self.broker_host}:{self.broker_port}...")
            self.client.connect(self.broker_host, self.broker_port, 60)
            self.client.loop_start()
            
            # Wait for connection
            time.sleep(1)
            return True
            
        except Exception as e:
            print(f"✗ Error connecting to MQTT broker: {e}")
            return False
            
    def run_stream(self):
        """Main streaming loop"""
        print(f"\n{'='*60}")
        print(f"Starting MQTT stream at {self.stream_frequency}Hz")
        print(f"Grind pattern: toggle every {self.grind_toggle_interval}s")
        print(f"Select pattern: toggle every {self.select_toggle_interval}s")
        print(f"Setpoint pattern: 0->4095->0 every {self.setpoint_cycle_interval}s")
        print(f"Press Ctrl+C to stop")
        print(f"{'='*60}\n")
        
        self.start_time = time.time()
        self.running = True
        
        try:
            while self.running and not self.stop_event.is_set():
                loop_start = time.time()
                elapsed_time = loop_start - self.start_time
                
                # Create and send payload
                payload = self.create_payload(elapsed_time)
                
                result = self.client.publish(self.topic, payload, qos=0)
                
                if result.rc == mqtt.MQTT_ERR_SUCCESS:
                    self.message_count += 1
                else:
                    print(f"✗ Failed to publish message. Return code: {result.rc}")
                    
                # Print status
                self.print_status(elapsed_time, payload)
                
                # Calculate sleep time to maintain frequency
                loop_duration = time.time() - loop_start
                sleep_time = max(0, self.stream_interval - loop_duration)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    print(f"⚠ Warning: Loop took {loop_duration*1000:.1f}ms, "
                          f"target is {self.stream_interval*1000:.1f}ms")
                    
        except KeyboardInterrupt:
            print(f"\n\n{'='*60}")
            print("Stream stopped by user (Ctrl+C)")
        except Exception as e:
            print(f"\n✗ Error during streaming: {e}")
        finally:
            self.cleanup()
            
    def cleanup(self):
        """Clean shutdown"""
        self.running = False
        self.stop_event.set()
        
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
            
        if self.start_time:
            total_time = time.time() - self.start_time
            avg_frequency = self.message_count / total_time if total_time > 0 else 0
            
            print(f"{'='*60}")
            print(f"Stream Statistics:")
            print(f"Total runtime: {total_time:.1f} seconds")
            print(f"Total messages: {self.message_count}")
            print(f"Average frequency: {avg_frequency:.1f} Hz")
            print(f"Target frequency: {self.stream_frequency} Hz")
            print(f"{'='*60}")
            
    def signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        print(f"\nReceived signal {signum}, shutting down...")
        self.running = False
        self.stop_event.set()

def main():
    # Create streamer instance
    streamer = EETestStreamer()
    
    # Setup signal handler for graceful shutdown
    signal.signal(signal.SIGINT, streamer.signal_handler)
    signal.signal(signal.SIGTERM, streamer.signal_handler)
    
    # Setup MQTT connection
    if not streamer.setup_mqtt():
        print("Failed to setup MQTT connection. Exiting.")
        sys.exit(1)
        
    # Run the streaming loop
    streamer.run_stream()

if __name__ == "__main__":
    main()


# sudo cp /etc/mosquitto/mosquitto.conf /etc/mosquitto/mosquitto.conf.backup

# sudo tee /etc/mosquitto/conf.d/allow_remote.conf > /dev/null << 'EOF'
# # Allow connections from any IP address on port 1883
# listener 1883
# allow_anonymous true
# EOF