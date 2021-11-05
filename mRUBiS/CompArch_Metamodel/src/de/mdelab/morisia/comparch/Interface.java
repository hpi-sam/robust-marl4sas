/**
 */
package de.mdelab.morisia.comparch;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Interface</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * </p>
 * <ul>
 *   <li>{@link de.mdelab.morisia.comparch.Interface#getType <em>Type</em>}</li>
 * </ul>
 *
 * @see de.mdelab.morisia.comparch.ComparchPackage#getInterface()
 * @model abstract="true"
 * @generated
 */
public interface Interface extends ArchitecturalElement {
	/**
	 * Returns the value of the '<em><b>Type</b></em>' reference.
	 * It is bidirectional and its opposite is '{@link de.mdelab.morisia.comparch.InterfaceType#getInstances <em>Instances</em>}'.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Type</em>' reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Type</em>' reference.
	 * @see #setType(InterfaceType)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getInterface_Type()
	 * @see de.mdelab.morisia.comparch.InterfaceType#getInstances
	 * @model opposite="instances" required="true"
	 * @generated
	 */
	InterfaceType getType();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.Interface#getType <em>Type</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Type</em>' reference.
	 * @see #getType()
	 * @generated
	 */
	void setType(InterfaceType value);

} // Interface
