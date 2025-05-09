#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class RobustImageProcessor(Node):
    def __init__(self):
        super().__init__('robust_image_processor')
        
        # Initialisiere CV-Bridge mit Exception Handling
        self.bridge = CvBridge()
        
        # Parameter mit Default-Werten
        self.declare_parameters(
            namespace='',
            parameters=[
                ('scale_percent', 50),
                ('jpeg_quality', 75),
                ('enable_scaling', True),
                ('enable_compression', False)
            ]
        )
        
        # QoS Profile für Kamera-Daten
        self.qos_profile = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.QoSReliabilityPolicy.RELIABLE,  
            history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Subscriber mit QoS Einstellungen
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            qos_profile=self.qos_profile)
        
                # Publisher mit gleichem QoS-Profil
        self.pub_color = self.create_publisher(
            Image, 
            'image_raw_180turned', 
            qos_profile=self.qos_profile)
        
        self.pub_bw = self.create_publisher(
            Image, 
            'image_raw_180turned_BW', 
            qos_profile=self.qos_profile)

        self.get_logger().info("Node mit kompatiblen QoS-Einstellungen initialisiert")
        
        # Publisher für beide Topics
        self.pub_color = self.create_publisher(Image, 'image_raw_180turned', self.qos_profile)
        self.pub_bw = self.create_publisher(Image, 'image_raw_180turned_BW', self.qos_profile)

        self.get_logger().info("Node initialisiert mit stabilen Einstellungen")

    def safe_process_image(self, cv_image, is_bw=False):
        try:
            # Parameter einmalig auslesen
            scale = self.get_parameter('scale_percent').value / 100.0
            enable_scale = self.get_parameter('enable_scaling').value
            enable_jpeg = self.get_parameter('enable_compression').value
            
            # Skalierung
            if enable_scale and scale != 1.0:
                h, w = cv_image.shape[:2]
                cv_image = cv2.resize(
                    cv_image, 
                    (int(w*scale), int(h*scale)), 
                    interpolation=cv2.INTER_LINEAR
                )
            
            # JPEG Komprimierung
            if enable_jpeg:
                quality = min(100, max(0, self.get_parameter('jpeg_quality').value))
                success, jpeg_data = cv2.imencode(
                    '.jpg', 
                    cv_image, 
                    [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                )
                if not success:
                    raise ValueError("JPEG Encoding fehlgeschlagen")
                
                msg = Image()
                msg.encoding = 'jpeg'
                msg.data = jpeg_data.tobytes()
                msg.height = cv_image.shape[0]
                msg.width = cv_image.shape[1]
                msg.step = 0  # Für komprimierte Bilder irrelevant
                return msg
            
            # Unkomprimierte Ausgabe
            encoding = 'mono8' if is_bw else 'bgr8'
            return self.bridge.cv2_to_imgmsg(cv_image, encoding=encoding)
            
        except Exception as e:
            self.get_logger().error(f"Prozessfehler: {str(e)}", throttle_duration_sec=5)
            return None

    def image_callback(self, msg):
        try:
            # Originalverarbeitung mit Schutz vor Corrupted Images
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Rotation und BW-Konvertierung
            rotated = cv2.rotate(cv_image, cv2.ROTATE_180)
            bw_image = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
            
            # Parallele Verarbeitung mit Timeout-Schutz
            color_msg = self.safe_process_image(rotated.copy())
            bw_msg = self.safe_process_image(bw_image.copy(), is_bw=True)
            
            if color_msg:
                color_msg.header = msg.header
                self.pub_color.publish(color_msg)
            
            if bw_msg:
                bw_msg.header = msg.header
                self.pub_bw.publish(bw_msg)
                
        except Exception as e:
            self.get_logger().error(f"Callback Fehler: {str(e)}", throttle_duration_sec=5)

def main(args=None):
    rclpy.init(args=args)
    node = RobustImageProcessor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Node wird ordnungsgemäß beendet...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
