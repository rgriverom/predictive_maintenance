import json
import time

from confluent_kafka import Producer

from data_loading import load_test
from features import prepare_training_data

KAFKA_BOOTSTRAP_SERVERS = 'localhost:9092'
TOPIC = 'engine-telemetry'
ENGINE_UNIT = 24  # pick any unit_number present in test_FD001.txt
DELAY_SECONDS = 1.0


def main():
    producer = Producer({'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS})

    _, _, _, _, feature_cols = prepare_training_data()
    test, _ = load_test()
    engine_data = test[test['unit_number'] == ENGINE_UNIT].sort_values('time_in_cycles')

    if engine_data.empty:
        raise ValueError(f'No test data found for unit_number={ENGINE_UNIT}')

    print(f'Streaming {len(engine_data)} cycles for engine {ENGINE_UNIT} to topic "{TOPIC}"...')

    for _, row in engine_data.iterrows():
        message = {
            'unit_number': int(row['unit_number']),
            'time_in_cycles': int(row['time_in_cycles']),
            'readings': {col: float(row[col]) for col in feature_cols},
        }
        producer.produce(TOPIC, value=json.dumps(message).encode('utf-8'))
        producer.poll(0)
        print(f'  sent cycle {message["time_in_cycles"]}')
        time.sleep(DELAY_SECONDS)

    producer.flush()
    print('Done streaming.')


if __name__ == '__main__':
    main()