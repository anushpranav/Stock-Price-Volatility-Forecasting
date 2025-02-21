import mysql.connector
from mysql.connector import Error
import bcrypt
from typing import Optional, Tuple, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, host: str, port: int, database: str, user: str, password: str):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.connection = None
        self.setup_database()

    def connect(self) -> None:
        """Establish database connection"""
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database
            )
            logger.info("Successfully connected to the database")
        except Error as e:
            logger.error(f"Error connecting to MySQL: {e}")
            raise

    def setup_database(self) -> None:
        """Set up the database and create required tables"""
        try:
            # Connect to MySQL server without selecting a database
            temp_conn = mysql.connector.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password
            )
            cursor = temp_conn.cursor()

            # Create database if it doesn't exist
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.database}")
            cursor.execute(f"USE {self.database}")

            # Create users table
            create_users_table = """
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                email VARCHAR(100) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP NULL
            )
            """
            cursor.execute(create_users_table)
            temp_conn.commit()
            cursor.close()
            temp_conn.close()

            # Establish connection to the created database
            self.connect()
            logger.info("Database setup completed successfully")

        except Error as e:
            logger.error(f"Error setting up database: {e}")
            raise

    @staticmethod
    def hash_password(password: str) -> bytes:
        """Hash a password using bcrypt"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt)

    @staticmethod
    def verify_password(password: str, password_hash: bytes) -> bool:
        """Verify a password against its hash"""
        return bcrypt.checkpw(password.encode('utf-8'), password_hash)

    def register_user(self, username: str, email: str, password: str) -> Tuple[bool, str]:
        """Register a new user"""
        try:
            cursor = self.connection.cursor()
            
            # Check if username or email already exists
            cursor.execute("SELECT username, email FROM users WHERE username = %s OR email = %s", 
                         (username, email))
            existing = cursor.fetchone()
            
            if existing:
                return False, "Username or email already exists"

            # Hash password and insert new user
            password_hash = self.hash_password(password)
            insert_query = """
            INSERT INTO users (username, email, password_hash)
            VALUES (%s, %s, %s)
            """
            cursor.execute(insert_query, (username, email, password_hash))
            self.connection.commit()
            
            return True, "Registration successful"

        except Error as e:
            logger.error(f"Error registering user: {e}")
            return False, f"Registration failed: {str(e)}"
        finally:
            cursor.close()

    def login_user(self, username: str, password: str) -> Tuple[bool, str, Optional[Dict]]:
        """Authenticate a user and return user details if successful"""
        try:
            cursor = self.connection.cursor(dictionary=True)
            
            # Get user details
            cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
            user = cursor.fetchone()
            
            if not user:
                return False, "Invalid username or password", None

            # Verify password
            if not self.verify_password(password, user['password_hash'].encode('utf-8')):
                return False, "Invalid username or password", None

            # Update last login timestamp
            cursor.execute("UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = %s", 
                         (user['id'],))
            self.connection.commit()

            # Remove password hash from user details
            user.pop('password_hash', None)
            return True, "Login successful", user

        except Error as e:
            logger.error(f"Error during login: {e}")
            return False, f"Login failed: {str(e)}", None
        finally:
            cursor.close()

    def close(self) -> None:
        """Close the database connection"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("Database connection closed")