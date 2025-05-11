import sys
import argparse
import socket
import driver
from data_logger import DataLogger

if __name__ == '__main__':
    # Configure argument parser
    parser = argparse.ArgumentParser(description='Python client to connect to TORCS SCRC server.')
    parser.add_argument('track_type', type=int, help='Track type (1=road, 2=oval, 3=dirt)')
    parser.add_argument('--host', default='localhost', help='Server host ip')
    parser.add_argument('--port', type=int, default=3001, help='Server host port')
    parser.add_argument('--id', default='SCR', help='Bot ID')
    parser.add_argument('--max_episodes', type=int, default=1, help='Maximum number of learning episodes')
    parser.add_argument('--max_steps', type=int, default=0, help='Maximum number of steps (0 for no limit)')
    parser.add_argument('--track', help='Track name')
    parser.add_argument('--stage', type=int, default=3, help='Stage (0=WARMUP, 1=QUALIFYING, 2=RACE, 3=UNKNOWN)')

    arguments = parser.parse_args()

    # Map track type
    track_type_map = {1: 'road', 2: 'oval', 3: 'dirt'}

    print(f'\nConnecting to server: {arguments.host}:{arguments.port}')
    print(f'Bot ID: {arguments.id}')
    print(f'Max episodes: {arguments.max_episodes}')
    print(f'Max steps: {arguments.max_steps}')
    print(f'Track type: {track_type_map[arguments.track_type]}')
    print(f'Stage: {arguments.stage}')
    print('*********************************************')

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    except socket.error as msg:
        print('Could not create socket.')
        sys.exit(-1)

    sock.settimeout(1.0)
    shutdownClient = False
    curEpisode = 0
    verbose = True

    d = driver.Driver(arguments.stage)
    d.set_track_type(arguments.track_type)
    print(f"Driver initialized with track type: {track_type_map[arguments.track_type]}")

    while not shutdownClient:
        while True:
            print(f'Sending ID: {arguments.id}')
            buf = arguments.id + d.init()
            try:
                sock.sendto(buf.encode(), (arguments.host, arguments.port))
            except socket.error:
                print("Failed to send data. Exiting...")
                sys.exit(-1)

            try:
                buf, addr = sock.recvfrom(1000)
                buf = buf.decode()
            except socket.error:
                print("No response from server...")
                continue

            if '***identified***' in buf:
                print(f'Received: {buf}')
                d.logger = DataLogger(
                    track_type_map[arguments.track_type],
                    'warmup' if arguments.stage == 0 else
                    'qualifying' if arguments.stage == 1 else
                    'race' if arguments.stage == 2 else 'unknown'
                )
                break

        currentStep = 0
        while True:
            try:
                buf, addr = sock.recvfrom(1000)
                buf = buf.decode()
            except socket.error:
                print("No response from server...")
                continue

            if verbose:
                print(f'Received: {buf}')

            if '***shutdown***' in buf:
                d.onShutDown()
                shutdownClient = True
                print('Client Shutdown')
                break

            if '***restart***' in buf:
                d.onRestart()
                print('Client Restart')
                break

            currentStep += 1
            if currentStep != arguments.max_steps:
                buf = d.drive(buf)
            else:
                buf = '(meta 1)'

            if verbose:
                print(f'Sending: {buf}')

            try:
                sock.sendto(buf.encode(), (arguments.host, arguments.port))
            except socket.error:
                print("Failed to send data. Exiting...")
                sys.exit(-1)

        curEpisode += 1
        if curEpisode == arguments.max_episodes:
            shutdownClient = True

    sock.close()