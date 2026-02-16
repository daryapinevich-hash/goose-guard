class TurretController:
    def aim(self, pred_pos, depth_m):
        """Получает ГОТОВЫЕ координаты предсказания"""
        pan_angle = self.pos_to_pan(pred_pos[0], frame_width)
        tilt_angle = self.depth_to_tilt(depth_m)

        self.servo_pan.write(pan_angle)
        self.servo_tilt.write(tilt_angle)

        if depth_m < 10:  # 10м
            self.laser_on()
