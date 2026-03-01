import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from core.config import SMTP_HOST, SMTP_PORT, SMTP_EMAIL, SMTP_PASSWORD


async def send_otp_email(to_email: str, otp_code: str, user_name: str):
    """Send OTP email via SMTP. Fails silently in dev if SMTP not configured."""
    if not SMTP_EMAIL or not SMTP_PASSWORD:
        print(f"\n[DEV] OTP for {to_email}: {otp_code}\n")
        return

    subject = "Your Password Reset OTP - MediCare AI"
    html_body = f"""
    <div style="font-family:Arial,sans-serif;max-width:480px;margin:0 auto;
                padding:32px;border:1px solid #e5e7eb;border-radius:12px;">
      <div style="text-align:center;margin-bottom:24px;">
        <span style="font-size:32px;">🏥</span>
        <h2 style="color:#1d4ed8;margin:8px 0 0;">MediCare AI</h2>
      </div>
      <h3 style="color:#111827;">Password Reset Request</h3>
      <p style="color:#6b7280;">Hi {user_name},</p>
      <p style="color:#6b7280;">
        We received a request to reset your password. Use the OTP below:
      </p>
      <div style="background:#eff6ff;border:2px dashed #3b82f6;border-radius:8px;
                  padding:20px;text-align:center;margin:24px 0;">
        <span style="font-size:36px;font-weight:700;letter-spacing:8px;color:#1d4ed8;">
          {otp_code}
        </span>
      </div>
      <p style="color:#ef4444;font-size:13px;">⏰ This OTP expires in <strong>10 minutes</strong>.</p>
      <p style="color:#ef4444;font-size:13px;">🔒 Max <strong>5 attempts</strong> allowed.</p>
      <hr style="border:none;border-top:1px solid #e5e7eb;margin:24px 0;">
      <p style="color:#9ca3af;font-size:12px;">
        If you didn't request this, please ignore this email.
        Your account remains secure.
      </p>
    </div>
    """

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = SMTP_EMAIL
    msg["To"] = to_email
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(SMTP_EMAIL, SMTP_PASSWORD)
            server.sendmail(SMTP_EMAIL, to_email, msg.as_string())
    except Exception as e:
        print(f"[EMAIL ERROR] {e}")