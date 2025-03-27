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

arguments = parser.parse_args()

# Print summary
print(f'Connecting to server host ip: {arguments.host_ip} @ port: {arguments.host_port}')
print(f'Bot ID: {arguments.id}')
print(f'Maximum episodes: {arguments.max_episodes}')
print(f'Maximum steps: {arguments.max_steps}')
print(f'Track: {arguments.track}')
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

verbose = False

d = driver.Driver(arguments.stage)

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
            print(f'Received: {buf}')
            # Initialize logger when race starts
            d.logger = DataLogger(arguments.track or 'unknown', 
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
