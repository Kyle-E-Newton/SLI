import React from 'react'
import Webcam from "react-webcam";

export default class WebcamCapture extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            screenshot: null,
            tab: 0,
            start: true
        };
    }
    
    setRef = webcam => {
        this.webcam = webcam;
    };
      
    capture = () => {
        const screenshot = this.webcam.getScreenshot();
        this.setState({ screenshot })
    };

    timer = setInterval((start) => {
        this.capture();
    }, 2000);


      
    render() {
    const videoConstraints = {
        width: 1280,
        height: 720,
        facingMode: "user"
        };
      
    return (
        <div>
            <Webcam id="cam"
                audio={false}
                ref={this.setRef}
                screenshotFormat="image/jpeg"
                videoConstraints={videoConstraints}
                />
            <button id="button1" onClick={this.capture}>Capture photo</button>
        </div>
        );
    }
}

