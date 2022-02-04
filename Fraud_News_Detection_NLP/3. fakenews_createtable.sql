
SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for fakenews
-- ----------------------------
DROP TABLE IF EXISTS `fakenews`;
CREATE TABLE `fakenews` (
  `id` int NOT NULL AUTO_INCREMENT,
  `author` varchar(255) DEFAULT NULL,
  `title` varchar(255) DEFAULT NULL,
  `text` mediumtext,
  `prediction` varchar(255) DEFAULT NULL,
  `created_time` datetime(6) DEFAULT NULL ON UPDATE CURRENT_TIMESTAMP(6),
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=3 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

SET FOREIGN_KEY_CHECKS = 1;
