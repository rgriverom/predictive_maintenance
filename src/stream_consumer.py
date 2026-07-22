import json

import requests
from confluent_kafka import Consumer

KAFKA_BOOTSTRAP_SERVERS = 'localhost:9092'
TOPIC = 'engine-telemetry'
API_URL = 'http://127.0.0.1:8001/predict'


def main():
    consumer = Consumer({
        'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS,
        'group.id': 'rul-predictor',
        'auto.offset.reset': 'earliest',
    })
    consumer.subscribe([TOPIC])

    print(f'Listening on topic "{TOPIC}"... (Ctrl+C to stop)')
    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                # UNKNOWN_TOPIC_OR_PART here is expected/harmless for the
                # first few seconds before the producer creates the topic.
                print(f'Consumer error (non-fatal, continuing): {msg.error()}')
                continue

            message = json.loads(msg.value().decode('utf-8'))

            try:
                response = requests.post(API_URL, json={'readings': message['readings']}, timeout=5)
                response.raise_for_status()
                predicted_rul = response.json()['predicted_rul_cycles']
                print(
                    f'Engine {message["unit_number"]}, cycle {message["time_in_cycles"]} '
                    f'-> predicted RUL: {predicted_rul:.1f} cycles'
                )
            except requests.exceptions.RequestException as e:
                print(f'Could not reach the API at {API_URL} — is serve_model.py running? ({e})')
    except KeyboardInterrupt:
        print('Stopping consumer.')
    finally:
        consumer.close()


if __name__ == '__main__':
    main()