import React from 'react'
import Webcam from "react-webcam";

export default class WebcamCapture extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            screenshot: null,
            tab: 0
        };
    }
    
    setRef = webcam => {
        this.webcam = webcam;
    };
      
    capture = () => {
        const screenshot = this.webcam.getScreenshot();
        this.setState({ screenshot })
    };
      
    render() {
    const videoConstraints = {
        width: 1280,
        height: 720,
        facingMode: "user"
        };
      
    return (
        <div>
            <Webcam 
                audio={false}
                height={350}
                ref={this.setRef}
                screenshotFormat="image/jpeg"
                width={350}
                videoConstraints={videoConstraints}
                />
            <button onClick={this.capture}>Capture photo</button>
            {this.state.screenshot ? <img src ={this.state.screenshot} /> : null}
        </div>
        );
    }
}

