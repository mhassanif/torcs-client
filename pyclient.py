import sys
import argparse
import socket
import driver
from data_logger import DataLogger

if __name__ == '__main__':
    pass

# Configure the argument parser
parser = argparse.ArgumentParser(description='Python client to connect to the TORCS SCRC server.')

parser.add_argument('--host', action='store', dest='host_ip', default='localhost',
                    help='Host IP address (default: localhost)')
parser.add_argument('--port', action='store', type=int, dest='host_port', default=3001,
                    help='Host port number (default: 3001)')
parser.add_argument('--id', action='store', dest='id', default='SCR',
                    help='Bot ID (default: SCR)')
parser.add_argument('--maxEpisodes', action='store', dest='max_episodes', type=int, default=1,
                    help='Maximum number of learning episodes (default: 1)')
parser.add_argument('--maxSteps', action='store', dest='max_steps', type=int, default=0,
                    help='Maximum number of steps (default: 0)')
parser.add_argument('--track', action='store', dest='track', default=None,
                    help='Name of the track')
parser.add_argument('--stage', action='store', dest='stage', type=int, default=3,
                    help='Stage (0 - Warm-Up, 1 - Qualifying, 2 - Race, 3 - Unknown)')
parser.add_argument('track_type', type=int, choices=[1, 2, 3],
                    help='Track type (1=road, 2=oval, 3=dirt)')

# Print raw arguments for debugging
print("Raw command line arguments:", sys.argv)

arguments = parser.parse_args()

# Print parsed arguments for debugging
print("\nParsed arguments:")
print(f"Track type: {arguments.track_type}")
print(f"All arguments: {vars(arguments)}")

# Map track type number to track name
track_type_map = {
    1: 'road',
    2: 'oval',
    3: 'dirt'
}

# Print summary
print(f'\nConnecting to server host ip: {arguments.host_ip} @ port: {arguments.host_port}')
print(f'Bot ID: {arguments.id}')
print(f'Maximum episodes: {arguments.max_episodes}')
print(f'Maximum steps: {arguments.max_steps}')
print(f'Track type: {track_type_map[arguments.track_type]}')
print(f'Stage: {arguments.stage}')
print('*********************************************')

try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
except socket.error as msg:
    print('Could not make a socket.')
    sys.exit(-1)

# one second timeout
sock.settimeout(1.0)

shutdownClient = False
curEpisode = 0

verbose = True  # Enable verbose mode to see all messages

d = driver.Driver(arguments.stage)
print("\nDebug - Driver initialized with stage:", arguments.stage)
print("Debug - Track mapping:", d.track_mapping)
print("Debug - Track parameters:", d.track_params)

# Set the track type in the driver
d.current_track_type = track_type_map[arguments.track_type]
print(f"Using track type: {d.current_track_type}")

while not shutdownClient:
    while True:
        print(f'Sending id to server: {arguments.id}')
        buf = arguments.id + d.init()
        print(f'Sending init string to server: {buf}')
        
        try:
            sock.sendto(buf.encode(), (arguments.host_ip, arguments.host_port))
        except socket.error as msg:
            print("Failed to send data...Exiting...")
            sys.exit(-1)
            
        try:
            buf, addr = sock.recvfrom(1000)
            buf = buf.decode()
        except socket.error as msg:
            print("Didn't get response from server...")
    
        if '***identified***' in buf:
            print(f'Received identification message: {buf}')
            # Initialize logger when race starts
            d.logger = DataLogger(track_type_map[arguments.track_type], 
                                'warmup' if arguments.stage == 0 else 
                                'qualifying' if arguments.stage == 1 else 
                                'race' if arguments.stage == 2 else 'unknown')
            break

    currentStep = 0
    
    while True:
        # wait for an answer from server
        buf = None
        try:
            buf, addr = sock.recvfrom(1000)
            buf = buf.decode()
        except socket.error as msg:
            print("Didn't get response from server...")
        
        if verbose:
            print(f'Received: {buf}')
        
        if buf and '***shutdown***' in buf:
            d.onShutDown()
            shutdownClient = True
            print('Client Shutdown')
            break
        
        if buf and '***restart***' in buf:
            d.onRestart()
            print('Client Restart')
            break
        
        currentStep += 1
        if currentStep != arguments.max_steps:
            if buf:
                buf = d.drive(buf)
        else:
            buf = '(meta 1)'
        
        if verbose:
            print(f'Sending: {buf}')
        
        if buf:
            try:
                sock.sendto(buf.encode(), (arguments.host_ip, arguments.host_port))
            except socket.error as msg:
                print("Failed to send data...Exiting...")
                sys.exit(-1)
    
    curEpisode += 1
    
    if curEpisode == arguments.max_episodes:
        shutdownClient = True
        

sock.close()
