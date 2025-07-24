#!/usr/bin/env python3
"""
RabbitMQ utility functions for connection management and messaging.
These functions provide generic helpers for RabbitMQ operations that can be
reused across different components.
"""

import json
import time
from contextlib import contextmanager

import pika
from pika.exceptions import AMQPConnectionError, AMQPChannelError


@contextmanager
def open_channel(rabbit_conf, logger, client_name=None):
    """
    Context manager that yields a Pika channel, then cleans it up.
    
    Args:
        rabbit_conf: RabbitMQ configuration dictionary containing connection parameters
        logger: Logger instance for logging events
        client_name: Optional client name for the connection
        
    Yields:
        pika.channel.Channel: An open RabbitMQ channel
    """
    connection = robust_connect(rabbit_conf, logger, client_name=client_name)
    channel = connection.channel()
    try:
        yield channel
    finally:
        channel.close()
        connection.close()


def publish_message(rabbit_conf, logger, routing_key, message, client_name=None):
    """
    Ephemeral publish: opens a connection/channel, publishes, then closes.
    
    Args:
        rabbit_conf: RabbitMQ configuration dictionary containing connection parameters
        logger: Logger instance for logging events
        routing_key: The routing key for the message
        message: The message to publish (string or bytes)
        client_name: Optional client name for the connection
    """
    with open_channel(rabbit_conf, logger, client_name) as channel:
        channel.basic_publish(
            exchange=rabbit_conf["exchange_name"],
            routing_key=routing_key,
            body=message.encode("utf-8") if isinstance(message, str) else message
        )


def robust_consume(rabbit_conf, logger, queue_name, routing_key, on_message_callback, stop_flag_func):
    """
    Repeatedly connect + channel.start_consuming().
    If disconnected, retry. If 'stop_flag_func()' is True, exit.
    
    Args:
        rabbit_conf: RabbitMQ configuration dictionary containing connection parameters
        logger: Logger instance for logging events
        queue_name: Name of the queue to consume from
        routing_key: Routing key pattern for the queue binding
        on_message_callback: Callback function to handle received messages
        stop_flag_func: Function that returns True when consumption should stop
    """
    logger.info(f"Starting robust_consume on {queue_name} with key {routing_key}...")
    while not stop_flag_func():
        try:
            connection = robust_connect(rabbit_conf, logger, client_name=f"tc_consumer_{queue_name}")
            channel = connection.channel()

            channel.basic_consume(
                queue=queue_name,
                on_message_callback=on_message_callback,
                auto_ack=False
            )
            logger.info(f"Begin consuming commands on queue={queue_name}")
            channel.start_consuming()

        except (AMQPConnectionError, AMQPChannelError) as e:
            logger.warning(f"Consumer lost connection: {e}. Reconnecting in 1s...")
            time.sleep(1)
        except Exception as e:
            logger.error(f"Unexpected error in robust_consume: {e}. Retrying in 1s...")
            time.sleep(1)
        finally:
            try: 
                channel.close()
            except:
                pass
            try:
                connection.close()
            except:
                pass

        if stop_flag_func():
            logger.info(f"Stop flag detected. Exiting consume loop for {queue_name}.")
            break


def robust_connect(rabbit_conf, logger, client_name=None):
    """
    Attempt to connect to RabbitMQ with advanced parameters:
      - client_properties => custom "connection_name"
      - heartbeat => keep the connection alive
      - blocked_connection_timeout => how long to wait if RabbitMQ is blocking
      - max_retries => 0 means infinite
      - wait_seconds => delay between retries
      - product / information => additional metadata
      
    Args:
        rabbit_conf: RabbitMQ configuration dictionary containing connection parameters
        logger: Logger instance for logging events
        client_name: Optional client name for the connection
        
    Returns:
        pika.BlockingConnection: An established RabbitMQ connection
        
    Raises:
        AMQPConnectionError: If max retries are exceeded
    """
    if not client_name:
        client_name = rabbit_conf["client_name_default"]
    attempts = 0
    creds = pika.PlainCredentials(rabbit_conf["user"], rabbit_conf["pass"])
    client_props = {
        "connection_name": client_name,
        "product": rabbit_conf["product"],
        "information": rabbit_conf["information"]
    }

    while True:
        try:
            params = pika.ConnectionParameters(
                host=rabbit_conf["host"],
                credentials=creds,
                heartbeat=rabbit_conf["heartbeat"],
                blocked_connection_timeout=rabbit_conf["blocked_connection_timeout"],
                client_properties=client_props
            )
            conn = pika.BlockingConnection(params)
            return conn
        except (AMQPConnectionError, AMQPChannelError) as e:
            attempts += 1
            logger.warning(
                f"[robust_connect] Connection failed: {e} (attempt {attempts}). "
                f"Waiting {rabbit_conf['wait_seconds']}s."
            )
            time.sleep(rabbit_conf["wait_seconds"])
            if rabbit_conf["max_retries"] > 0 and attempts >= rabbit_conf["max_retries"]:
                raise
