import qrcode
from PIL import Image, ImageDraw, ImageFont

def generate_qr(data, filename="qr_code.png"):
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    img.save(filename)
    print(f"✅ QR Code salvato: {filename}")
    return filename

if __name__ == "__main__":
    generate_qr("monero:45...?amount=0.01", "payment_qr.png")
