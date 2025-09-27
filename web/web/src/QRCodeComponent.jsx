import React from "react";
import QRCode from "react-qr-code";

export default function QRCodeComponent({ value }) {
  return (
    <div className="qr-code-wrapper">
      <QRCode value={value} size={180} />
    </div>
  );
}
