// Código Arduino
void setup() {
  Serial.begin(115200); // Velocidad rápida
}

void loop() {
  // Leer los 3 sensores
  int val1 = analogRead(A0); // Cadera (Hip)
  int val2 = analogRead(A1); // Rodilla (Knee)
  int val3 = analogRead(A2); // Tobillo (Ankle)
  
  // Enviar en formato CSV: "1023,512,0"
  Serial.print(val1);
  Serial.print(",");
  Serial.print(val2);
  Serial.print(",");
  Serial.println(val3); // El último lleva println para el salto de línea
  
  delay(10); // Pequeña pausa para estabilidad
}
