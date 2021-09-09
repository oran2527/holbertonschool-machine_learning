-- creates a table users 2
CREATE TABLE IF NOT EXISTS
        users(id INTEGER NOT NULL AUTO_INCREMENT PRIMARY KEY,
              email VARCHAR(255) NOT NULL UNIQUE,
              name VARCHAR(255),
              country ENUM('US', 'CO', 'TN') NOT NULL DEFAULT 'US');
